#!/usr/bin/env python3
"""
Render Tesla dashcam video with a right-side telemetry panel and in-frame HUD.

This script adapts Tesla's public dashcam SEI extraction logic from:
https://github.com/teslamotors/dashcam

It reads SEI protobuf metadata embedded in the MP4, then streams frames through
ffmpeg and writes a new MP4 where:
* the original video remains unobstructed on the left
* a telemetry table is rendered in a separate panel on the right
* compact visual indicators are drawn inside the video itself
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import DecodeError
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # pragma: no cover - dependency availability varies by machine
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency availability varies by machine
    np = None

import av as _av


GEAR_NAMES = {
    0: "PARK",
    1: "DRIVE",
    2: "REVERSE",
    3: "NEUTRAL",
}

AUTOPILOT_NAMES = {
    0: "NONE",
    1: "SELF_DRIVING",
    2: "AUTOSTEER",
    3: "TACC",
}

DEFAULT_IMAGE_PARAMS_PATH = Path(__file__).with_name("default_image_params.json")


@dataclass
class TelemetryFrame:
    version: int = 0
    gear_state: int = 0
    frame_seq_no: int = 0
    vehicle_speed_mps: float = 0.0
    accelerator_pedal_position: float = 0.0
    steering_wheel_angle: float = 0.0
    blinker_on_left: bool = False
    blinker_on_right: bool = False
    brake_applied: bool = False
    autopilot_state: int = 0
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    heading_deg: float = 0.0
    linear_acceleration_mps2_x: float = 0.0
    linear_acceleration_mps2_y: float = 0.0
    linear_acceleration_mps2_z: float = 0.0

    @property
    def gear_label(self) -> str:
        return GEAR_NAMES.get(self.gear_state, str(self.gear_state))

    @property
    def autopilot_label(self) -> str:
        return AUTOPILOT_NAMES.get(self.autopilot_state, str(self.autopilot_state))

    @property
    def g_force(self) -> float:
        g = math.sqrt(
            self.linear_acceleration_mps2_x ** 2
            + self.linear_acceleration_mps2_y ** 2
            + self.linear_acceleration_mps2_z ** 2
        )
        return g / 9.80665


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: Optional[int]


class TelemetryTimeline:
    def __init__(
        self,
        items: Sequence[Optional[TelemetryFrame]],
        expected_frames: Optional[int],
        offset_frames: int = 0,
    ) -> None:
        self.items = list(items)
        self.expected_frames = expected_frames
        self.offset_frames = offset_frames
        # Tesla's viewer binds each pending SEI packet directly to the next
        # slice/IDR frame. Keep that frame-by-frame mapping intact here instead
        # of stretching the telemetry timeline to match ffprobe's frame count.
        self.alignment_mode = "direct"

    def __len__(self) -> int:
        return len(self.items)

    @property
    def frame_count_delta(self) -> Optional[int]:
        if self.expected_frames is None:
            return None
        return len(self.items) - self.expected_frames

    def frame_at(self, frame_index: int) -> Optional[TelemetryFrame]:
        if not self.items:
            raise ValueError("No SEI telemetry frames available in this MP4.")

        adjusted_index = frame_index + self.offset_frames
        return self.items[max(0, min(adjusted_index, len(self.items) - 1))]


def build_sei_metadata_class():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "dashcam.proto"
    file_proto.syntax = "proto3"

    message = file_proto.message_type.add()
    message.name = "SeiMetadata"

    gear_enum = message.enum_type.add()
    gear_enum.name = "Gear"
    for name, number in (
        ("GEAR_PARK", 0),
        ("GEAR_DRIVE", 1),
        ("GEAR_REVERSE", 2),
        ("GEAR_NEUTRAL", 3),
    ):
        value = gear_enum.value.add()
        value.name = name
        value.number = number

    autopilot_enum = message.enum_type.add()
    autopilot_enum.name = "AutopilotState"
    for name, number in (
        ("NONE", 0),
        ("SELF_DRIVING", 1),
        ("AUTOSTEER", 2),
        ("TACC", 3),
    ):
        value = autopilot_enum.value.add()
        value.name = name
        value.number = number

    add_field(message, "version", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add_field(
        message,
        "gear_state",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        type_name=".SeiMetadata.Gear",
    )
    add_field(message, "frame_seq_no", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_field(message, "vehicle_speed_mps", 4, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    add_field(
        message,
        "accelerator_pedal_position",
        5,
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
    )
    add_field(
        message,
        "steering_wheel_angle",
        6,
        descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
    )
    add_field(message, "blinker_on_left", 7, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    add_field(message, "blinker_on_right", 8, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    add_field(message, "brake_applied", 9, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL)
    add_field(
        message,
        "autopilot_state",
        10,
        descriptor_pb2.FieldDescriptorProto.TYPE_ENUM,
        type_name=".SeiMetadata.AutopilotState",
    )
    add_field(message, "latitude_deg", 11, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    add_field(message, "longitude_deg", 12, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    add_field(message, "heading_deg", 13, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    add_field(
        message,
        "linear_acceleration_mps2_x",
        14,
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )
    add_field(
        message,
        "linear_acceleration_mps2_y",
        15,
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )
    add_field(
        message,
        "linear_acceleration_mps2_z",
        16,
        descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE,
    )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("SeiMetadata")
    return message_factory.GetMessageClass(descriptor)


def add_field(
    message: descriptor_pb2.DescriptorProto,
    name: str,
    number: int,
    field_type: int,
    type_name: Optional[str] = None,
) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = field_type
    if type_name:
        field.type_name = type_name


def find_mdat(fp: BinaryIO) -> Tuple[int, int]:
    fp.seek(0)
    while True:
        header = fp.read(8)
        if len(header) < 8:
            raise RuntimeError("mdat atom not found")
        size32, atom_type = struct.unpack(">I4s", header)
        if size32 == 1:
            large = fp.read(8)
            if len(large) != 8:
                raise RuntimeError("truncated extended atom size")
            atom_size = struct.unpack(">Q", large)[0]
            header_size = 16
        else:
            atom_size = size32 if size32 else 0
            header_size = 8
        if atom_type == b"mdat":
            payload_size = atom_size - header_size if atom_size else 0
            return fp.tell(), payload_size
        if atom_size < header_size:
            raise RuntimeError("invalid MP4 atom size")
        fp.seek(atom_size - header_size, 1)


# ── MP4 box navigation (for ctts/stts B-frame detection) ────────────────────

_CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta", b"dinf"}


def iter_mp4_boxes(
    fp: BinaryIO, start: int, end: int,
) -> Generator[Tuple[bytes, int, int], None, None]:
    """Yield (box_type, data_offset, data_size) for boxes within [start, end)."""
    pos = start
    while pos + 8 <= end:
        fp.seek(pos)
        header = fp.read(8)
        if len(header) < 8:
            break
        size32, box_type = struct.unpack(">I4s", header)
        if size32 == 1:
            ext = fp.read(8)
            if len(ext) < 8:
                break
            box_size = struct.unpack(">Q", ext)[0]
            header_len = 16
        elif size32 == 0:
            box_size = end - pos
            header_len = 8
        else:
            box_size = size32
            header_len = 8
        if box_size < header_len:
            break
        data_offset = pos + header_len
        data_size = box_size - header_len
        yield box_type, data_offset, data_size
        pos += box_size


def find_mp4_box_path(
    fp: BinaryIO, path: Sequence[bytes], start: int, end: int,
) -> Optional[Tuple[int, int]]:
    """Walk a box path like (b"moov", b"trak", b"mdia", ...) returning (data_offset, data_size)."""
    target = path[0]
    for box_type, data_offset, data_size in iter_mp4_boxes(fp, start, end):
        if box_type != target:
            continue
        if len(path) == 1:
            return data_offset, data_size
        if box_type in _CONTAINER_BOXES:
            result = find_mp4_box_path(fp, path[1:], data_offset, data_offset + data_size)
            if result is not None:
                return result
    return None


def find_video_stbl(fp: BinaryIO) -> Optional[Tuple[int, int]]:
    """Locate the stbl box for the first video track."""
    fp.seek(0, 2)
    file_end = fp.tell()
    moov = find_mp4_box_path(fp, (b"moov",), 0, file_end)
    if moov is None:
        return None
    moov_off, moov_sz = moov
    moov_end = moov_off + moov_sz
    for box_type, data_offset, data_size in iter_mp4_boxes(fp, moov_off, moov_end):
        if box_type != b"trak":
            continue
        trak_end = data_offset + data_size
        hdlr = find_mp4_box_path(fp, (b"mdia", b"hdlr"), data_offset, trak_end)
        if hdlr is None:
            continue
        hdlr_off, hdlr_sz = hdlr
        fp.seek(hdlr_off)
        hdlr_data = fp.read(min(hdlr_sz, 12))
        if len(hdlr_data) >= 12 and hdlr_data[8:12] == b"vide":
            stbl = find_mp4_box_path(
                fp, (b"mdia", b"minf", b"stbl"), data_offset, trak_end,
            )
            if stbl is not None:
                return stbl
    return None


def parse_stts_box(fp: BinaryIO, offset: int, size: int) -> List[int]:
    """Parse stts (sample-to-time) box into per-sample durations."""
    fp.seek(offset)
    header = fp.read(min(size, 8))
    if len(header) < 8:
        return []
    entry_count = struct.unpack(">I", header[4:8])[0]
    durations: List[int] = []
    for _ in range(entry_count):
        entry = fp.read(8)
        if len(entry) < 8:
            break
        sample_count, sample_delta = struct.unpack(">II", entry)
        durations.extend([sample_delta] * sample_count)
    return durations


def parse_ctts_box(fp: BinaryIO, offset: int, size: int) -> List[int]:
    """Parse ctts (composition time to sample) box into per-sample offsets."""
    fp.seek(offset)
    header = fp.read(min(size, 8))
    if len(header) < 8:
        return []
    version = header[0]
    entry_count = struct.unpack(">I", header[4:8])[0]
    offsets: List[int] = []
    for _ in range(entry_count):
        entry = fp.read(8)
        if len(entry) < 8:
            break
        sample_count = struct.unpack(">I", entry[0:4])[0]
        sample_offset = struct.unpack(">i" if version else ">I", entry[4:8])[0]
        offsets.extend([sample_offset] * sample_count)
    return offsets


def reorder_telemetry_to_display_order(
    aligned: List[Optional[TelemetryFrame]],
    mp4_path: Path,
) -> Tuple[List[Optional[TelemetryFrame]], bool]:
    """
    Reorder a decode-order telemetry list to display (PTS) order.

    Returns (reordered_list, had_bframes).
    If no ctts box exists or all offsets are equal, returns the list unchanged.
    """
    with mp4_path.open("rb") as fp:
        stbl = find_video_stbl(fp)
        if stbl is None:
            return aligned, False
        stbl_off, stbl_sz = stbl
        stbl_end = stbl_off + stbl_sz

        ctts_loc = find_mp4_box_path(fp, (b"ctts",), stbl_off, stbl_end)
        if ctts_loc is None:
            return aligned, False
        ctts_offsets = parse_ctts_box(fp, *ctts_loc)
        if not ctts_offsets or len(set(ctts_offsets)) <= 1:
            return aligned, False

        stts_loc = find_mp4_box_path(fp, (b"stts",), stbl_off, stbl_end)
        if stts_loc is None:
            return aligned, False
        stts_durations = parse_stts_box(fp, *stts_loc)

    n = min(len(aligned), len(stts_durations), len(ctts_offsets))
    if n == 0:
        return aligned, False

    dts = 0
    pts_indexed: List[Tuple[int, int]] = []
    for i in range(n):
        pts_indexed.append((dts + ctts_offsets[i], i))
        dts += stts_durations[i]

    pts_indexed.sort()
    reordered: List[Optional[TelemetryFrame]] = [aligned[idx] for _, idx in pts_indexed]
    if len(aligned) > n:
        reordered.extend(aligned[n:])
    return reordered, True


def iter_nals(fp: BinaryIO, offset: int, size: int) -> Generator[bytes, None, None]:
    nal_id_sei = 6
    nal_sei_user_data = 5

    fp.seek(offset)
    consumed = 0
    while size == 0 or consumed < size:
        header = fp.read(4)
        if len(header) < 4:
            break
        nal_size = struct.unpack(">I", header)[0]
        if nal_size < 2:
            fp.seek(nal_size, 1)
            consumed += 4 + nal_size
            continue

        first_two = fp.read(2)
        if len(first_two) != 2:
            break

        if (first_two[0] & 0x1F) != nal_id_sei or first_two[1] != nal_sei_user_data:
            fp.seek(nal_size - 2, 1)
            consumed += 4 + nal_size
            continue

        rest = fp.read(nal_size - 2)
        if len(rest) != nal_size - 2:
            break
        consumed += 4 + nal_size
        yield first_two + rest


def extract_proto_payload(nal: bytes) -> Optional[bytes]:
    if len(nal) < 2:
        return None
    for index in range(3, len(nal) - 1):
        current = nal[index]
        if current == 0x42:
            continue
        if current == 0x69 and index > 2:
            return strip_emulation_prevention_bytes(nal[index + 1 : -1])
        break
    return None


def strip_emulation_prevention_bytes(data: bytes) -> bytes:
    stripped = bytearray()
    zero_count = 0
    for byte in data:
        if zero_count >= 2 and byte == 0x03:
            zero_count = 0
            continue
        stripped.append(byte)
        zero_count = 0 if byte != 0 else zero_count + 1
    return bytes(stripped)


def read_telemetry(mp4_path: Path) -> List[TelemetryFrame]:
    sei_class = build_sei_metadata_class()
    frames: List[TelemetryFrame] = []
    with mp4_path.open("rb") as handle:
        offset, size = find_mdat(handle)
        for nal in iter_nals(handle, offset, size):
            payload = extract_proto_payload(nal)
            if not payload:
                continue
            message = sei_class()
            try:
                message.ParseFromString(payload)
            except DecodeError:
                continue
            frames.append(to_telemetry_frame(message))
    return frames


def iter_mp4_nals(fp: BinaryIO, offset: int, size: int) -> Generator[Tuple[int, bytes], None, None]:
    fp.seek(offset)
    consumed = 0
    while size == 0 or consumed < size:
        header = fp.read(4)
        if len(header) < 4:
            break
        nal_size = struct.unpack(">I", header)[0]
        if nal_size < 1:
            consumed += 4 + max(nal_size, 0)
            continue
        payload = fp.read(nal_size)
        if len(payload) != nal_size:
            break
        consumed += 4 + nal_size
        nal_type = payload[0] & 0x1F
        yield nal_type, payload


def read_frame_aligned_telemetry(mp4_path: Path) -> List[Optional[TelemetryFrame]]:
    """
    Direct port of Tesla's DashcamMP4.parseFrames() telemetry binding.

    Walks all NAL units in mdat, attaching the pending SEI to each slice/IDR
    frame — exactly as Tesla's JavaScript viewer does.  Every slice/IDR NAL
    produces one entry (Tesla does the same; dashcam files are single-slice).
    """
    sei_class = build_sei_metadata_class()
    aligned: List[Optional[TelemetryFrame]] = []
    pending: Optional[TelemetryFrame] = None

    with mp4_path.open("rb") as handle:
        offset, size = find_mdat(handle)
        for nal_type, nal in iter_mp4_nals(handle, offset, size):
            if nal_type == 6:
                payload = extract_proto_payload(nal)
                if not payload:
                    continue
                message = sei_class()
                try:
                    message.ParseFromString(payload)
                except DecodeError:
                    continue
                pending = to_telemetry_frame(message)
            elif nal_type in (1, 5):
                aligned.append(pending)
                pending = None

    return aligned


def to_telemetry_frame(message) -> TelemetryFrame:
    values = {field.name: value for field, value in message.ListFields()}
    return TelemetryFrame(
        version=values.get("version", 0),
        gear_state=values.get("gear_state", 0),
        frame_seq_no=values.get("frame_seq_no", 0),
        vehicle_speed_mps=values.get("vehicle_speed_mps", 0.0),
        accelerator_pedal_position=values.get("accelerator_pedal_position", 0.0),
        steering_wheel_angle=values.get("steering_wheel_angle", 0.0),
        blinker_on_left=values.get("blinker_on_left", False),
        blinker_on_right=values.get("blinker_on_right", False),
        brake_applied=values.get("brake_applied", False),
        autopilot_state=values.get("autopilot_state", 0),
        latitude_deg=values.get("latitude_deg", 0.0),
        longitude_deg=values.get("longitude_deg", 0.0),
        heading_deg=values.get("heading_deg", 0.0),
        linear_acceleration_mps2_x=values.get("linear_acceleration_mps2_x", 0.0),
        linear_acceleration_mps2_y=values.get("linear_acceleration_mps2_y", 0.0),
        linear_acceleration_mps2_z=values.get("linear_acceleration_mps2_z", 0.0),
    )


def probe_video(mp4_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-of",
        "json",
        str(mp4_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1")
    frame_count = None
    if stream.get("nb_frames"):
        try:
            frame_count = int(stream["nb_frames"])
        except ValueError:
            frame_count = None
    elif stream.get("duration"):
        frame_count = int(round(float(stream["duration"]) * fps))
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def parse_fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 30.0
    return float(numerator) / denominator_value


def pick_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_candidates = []
    if bold:
        font_candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        font_candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in font_candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        panel_width: int,
        speed_unit: str,
        hud_position: str,
        hud_transparency: float,
    ) -> None:
        self.video_width = width
        self.video_height = height
        self.panel_width = panel_width
        self.speed_unit = speed_unit
        self.hud_position = hud_position
        self.hud_transparency = max(0.0, min(1.0, hud_transparency))
        self.canvas_width = width + panel_width
        self.canvas_height = height

        self.font_hud_speed = pick_font(max(56, int(height * 0.058)), bold=True)
        self.font_hud_unit = pick_font(max(18, int(height * 0.020)), bold=True)
        self.font_hud_gear = pick_font(max(14, int(height * 0.015)), bold=True)
        self.font_hud_label = pick_font(max(12, int(height * 0.013)), bold=True)
        self.font_hud_value = pick_font(max(17, int(height * 0.019)), bold=True)
        self.font_panel_label = pick_font(max(15, int(height * 0.0175)), bold=True)
        self.font_panel_value = pick_font(max(19, int(height * 0.0215)), bold=True)

    def render(self, frame: Image.Image, telemetry: Optional[TelemetryFrame]) -> Image.Image:
        canvas = Image.new("RGB", (self.canvas_width, self.canvas_height), (9, 10, 15))
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")

        if telemetry is not None:
            self.draw_video_hud(draw, telemetry)
        self.draw_side_panel(draw, telemetry)
        return canvas

    def draw_video_hud(self, draw: ImageDraw.ImageDraw, telemetry: TelemetryFrame) -> None:
        margin_x = 22
        margin_y = 18
        card_w = min(390, max(290, int(self.video_width * 0.27)))
        card_h = min(172, max(142, int(self.video_height * 0.17)))
        hud_left, hud_top = self.hud_origin(card_w, card_h, margin_x, margin_y)
        box = (hud_left, hud_top, hud_left + card_w, hud_top + card_h)
        alpha = max(0, min(255, int(round(255 * (1.0 - self.hud_transparency)))))

        # Soft drop shadow
        draw.rounded_rectangle(
            (box[0] + 2, box[1] + 3, box[2] + 2, box[3] + 3),
            radius=18, fill=(0, 0, 0, max(1, alpha // 4)),
        )
        # Card: warm dark glass
        draw.rounded_rectangle(
            box, radius=18,
            fill=(22, 24, 34, alpha),
            outline=(255, 255, 255, 15),
            width=1,
        )

        pad = 14
        lx = box[0] + pad
        rx = box[2] - pad

        # ── Row 1: Gear pill + Steering wheel icon + angle ──
        row1_y = box[1] + pad
        gear = telemetry.gear_label
        gear_colors = {
            0: (58, 62, 74, 185),
            1: (50, 125, 220, 195),
            2: (200, 72, 68, 195),
            3: (170, 158, 48, 195),
        }
        pill_bg = gear_colors.get(telemetry.gear_state, (58, 62, 74, 185))
        gw = measure_text(self.font_hud_gear, gear)[0] + 16
        gh = measure_text(self.font_hud_gear, gear)[1] + 8
        draw.rounded_rectangle((lx, row1_y, lx + gw, row1_y + gh), radius=7, fill=pill_bg)
        self.draw_text(
            draw, (lx + 8, row1_y + 3), gear,
            self.font_hud_gear, (255, 255, 255, 240), stroke_fill=(0, 0, 0, 40),
        )

        steer = telemetry.steering_wheel_angle
        steer_text = f"{abs(steer):.0f}\u00b0"
        stw = measure_text(self.font_hud_value, steer_text)[0]
        self.draw_text(
            draw, (rx - stw, row1_y + 1), steer_text,
            self.font_hud_value, (190, 200, 215, 235), stroke_fill=(0, 0, 0, 40),
        )
        wheel_r = 11
        wheel_cx = rx - stw - wheel_r - 8
        wheel_cy = row1_y + gh // 2
        wheel_color = (170, 180, 198, 190)
        draw.ellipse(
            (wheel_cx - wheel_r, wheel_cy - wheel_r,
             wheel_cx + wheel_r, wheel_cy + wheel_r),
            outline=wheel_color, width=2,
        )
        rot = math.radians(max(-180.0, min(180.0, steer)))
        draw.line(
            (wheel_cx, wheel_cy,
             wheel_cx + math.sin(rot) * wheel_r * 0.65,
             wheel_cy - math.cos(rot) * wheel_r * 0.65),
            fill=wheel_color, width=2,
        )
        draw.arc(
            (wheel_cx - wheel_r + 3, wheel_cy - wheel_r + 3,
             wheel_cx + wheel_r - 3, wheel_cy + wheel_r - 3),
            215, 325, fill=wheel_color, width=2,
        )

        # ── Row 2: Speed (hero, centered) ──
        speed_val, speed_lbl = format_speed(telemetry.vehicle_speed_mps, self.speed_unit)
        spd_w, spd_h = measure_text(self.font_hud_speed, speed_val)
        unit_w, unit_h = measure_text(self.font_hud_unit, speed_lbl)
        total_w = spd_w + 6 + unit_w
        spd_x = box[0] + (card_w - total_w) // 2
        spd_y = box[1] + (card_h - spd_h) // 2 - 4
        self.draw_text(
            draw, (spd_x, spd_y), speed_val,
            self.font_hud_speed, (250, 252, 255, 255),
            stroke_fill=(0, 0, 0, 90), stroke_width=2,
        )
        self.draw_text(
            draw, (spd_x + spd_w + 6, spd_y + spd_h - unit_h - 3), speed_lbl,
            self.font_hud_unit, (135, 145, 162, 225), stroke_fill=(0, 0, 0, 35),
        )

        # ── Subtle divider ──
        div_y = box[3] - pad - 24
        draw.line((lx + 4, div_y, rx - 4, div_y), fill=(255, 255, 255, 14), width=1)

        # ── Row 3: Blinker pills + Brake + ACC bar + G-force ──
        row3_y = div_y + 5
        row3_cy = row3_y + 9

        blink_w, blink_h = 20, 16
        l_on = telemetry.blinker_on_left
        r_on = telemetry.blinker_on_right
        l_bg = (255, 172, 40, 230) if l_on else (48, 52, 62, 130)
        r_bg = (255, 172, 40, 230) if r_on else (48, 52, 62, 130)
        arrow_on_c = (255, 255, 255, 240)
        arrow_off_c = (72, 78, 90, 150)

        # Left blinker pill + arrow
        if l_on:
            draw.rounded_rectangle(
                (lx - 2, row3_cy - blink_h // 2 - 2,
                 lx + blink_w + 2, row3_cy + blink_h // 2 + 2),
                radius=7, fill=(255, 172, 40, 35),
            )
        draw.rounded_rectangle(
            (lx, row3_cy - blink_h // 2, lx + blink_w, row3_cy + blink_h // 2),
            radius=5, fill=l_bg,
        )
        draw.polygon([
            (lx + 4, row3_cy),
            (lx + 12, row3_cy - 4),
            (lx + 12, row3_cy + 4),
        ], fill=arrow_on_c if l_on else arrow_off_c)

        # Right blinker pill + arrow
        rbx = lx + blink_w + 5
        if r_on:
            draw.rounded_rectangle(
                (rbx - 2, row3_cy - blink_h // 2 - 2,
                 rbx + blink_w + 2, row3_cy + blink_h // 2 + 2),
                radius=7, fill=(255, 172, 40, 35),
            )
        draw.rounded_rectangle(
            (rbx, row3_cy - blink_h // 2, rbx + blink_w, row3_cy + blink_h // 2),
            radius=5, fill=r_bg,
        )
        draw.polygon([
            (rbx + blink_w - 4, row3_cy),
            (rbx + blink_w - 12, row3_cy - 4),
            (rbx + blink_w - 12, row3_cy + 4),
        ], fill=arrow_on_c if r_on else arrow_off_c)

        # Brake pill
        brk_x = rbx + blink_w + 8
        brk_w, brk_h = 28, 18
        brk_active = telemetry.brake_applied
        brk_bg = (210, 68, 62, 205) if brk_active else (42, 46, 56, 125)
        if brk_active:
            draw.rounded_rectangle(
                (brk_x - 2, row3_cy - brk_h // 2 - 2,
                 brk_x + brk_w + 2, row3_cy + brk_h // 2 + 2),
                radius=8, fill=(210, 68, 62, 30),
            )
        draw.rounded_rectangle(
            (brk_x, row3_cy - brk_h // 2, brk_x + brk_w, row3_cy + brk_h // 2),
            radius=6, fill=brk_bg,
        )
        btw = measure_text(self.font_hud_label, "B")[0]
        btc = (255, 255, 255, 245) if brk_active else (72, 78, 90, 170)
        self.draw_text(
            draw, (brk_x + (brk_w - btw) // 2, row3_cy - brk_h // 2 + 2), "B",
            self.font_hud_label, btc, stroke_fill=(0, 0, 0, 25),
        )

        # ACC progress bar
        acc_pct = max(0.0, min(100.0, telemetry.accelerator_pedal_position))
        bar_x = brk_x + brk_w + 10
        bar_w = 55
        bar_h = 8
        bar_y = row3_cy - bar_h // 2
        draw.rounded_rectangle(
            (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h),
            radius=4, fill=(36, 40, 50, 160),
        )
        if acc_pct > 0:
            fill_w = max(4, int(bar_w * acc_pct / 100.0))
            bar_color = (38, 195, 135, 235) if acc_pct < 80 else (255, 172, 40, 235)
            draw.rounded_rectangle(
                (bar_x, bar_y, bar_x + fill_w, bar_y + bar_h),
                radius=4, fill=bar_color,
            )
        acc_label = f"{acc_pct:.0f}%"
        alw, alh = measure_text(self.font_hud_value, acc_label)
        self.draw_text(
            draw, (bar_x + bar_w + 5, row3_cy - alh // 2), acc_label,
            self.font_hud_value, (190, 200, 215, 230), stroke_fill=(0, 0, 0, 30),
        )

        # G-force circle with moving dot
        gf_r = 13
        gf_cx = rx - gf_r - 2
        gf_cy = row3_cy
        draw.ellipse(
            (gf_cx - gf_r, gf_cy - gf_r, gf_cx + gf_r, gf_cy + gf_r),
            outline=(155, 165, 182, 120), width=1,
        )
        draw.line(
            (gf_cx - gf_r + 2, gf_cy, gf_cx + gf_r - 2, gf_cy),
            fill=(68, 75, 88, 70), width=1,
        )
        draw.line(
            (gf_cx, gf_cy - gf_r + 2, gf_cx, gf_cy + gf_r - 2),
            fill=(68, 75, 88, 70), width=1,
        )
        scale = gf_r / 5.0
        dot_x = gf_cx + telemetry.linear_acceleration_mps2_x * scale
        dot_y = gf_cy - telemetry.linear_acceleration_mps2_y * scale
        dot_x = max(gf_cx - gf_r + 4, min(gf_cx + gf_r - 4, dot_x))
        dot_y = max(gf_cy - gf_r + 4, min(gf_cy + gf_r - 4, dot_y))
        draw.ellipse((dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5), fill=(50, 190, 200, 45))
        draw.ellipse((dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3), fill=(55, 195, 210, 240))
        g_text = f"{telemetry.g_force:.1f}g"
        gtw, gth = measure_text(self.font_hud_value, g_text)
        self.draw_text(
            draw, (gf_cx - gf_r - gtw - 5, gf_cy - gth // 2), g_text,
            self.font_hud_value, (155, 165, 182, 220), stroke_fill=(0, 0, 0, 25),
        )

    def hud_origin(
        self,
        card_width: int,
        card_height: int,
        margin_x: int,
        margin_y: int,
    ) -> Tuple[int, int]:
        if self.hud_position == "top-left":
            return margin_x, margin_y
        if self.hud_position == "bottom-left":
            return margin_x, self.video_height - card_height - margin_y
        if self.hud_position == "bottom-right":
            return self.video_width - card_width - margin_x, self.video_height - card_height - margin_y
        return self.video_width - card_width - margin_x, margin_y

    def draw_side_panel(self, draw: ImageDraw.ImageDraw, telemetry: Optional[TelemetryFrame]) -> None:
        panel_left = self.video_width
        draw.rectangle((panel_left, 0, self.canvas_width, self.canvas_height), fill=(15, 18, 24, 255))

        x = panel_left + 26
        y = 26

        label_height = measure_text(self.font_panel_label, "Hg")[1]
        value_height = measure_text(self.font_panel_value, "Hg")[1]
        label_gap = max(4, int(self.video_height * 0.004))
        row_gap = max(10, int(self.video_height * 0.008))

        for label, value in panel_rows(telemetry):
            self.draw_text(
                draw,
                (x, y),
                label,
                self.font_panel_label,
                (176, 180, 188, 255),
                stroke_fill=(0, 0, 0, 80),
            )
            y += label_height + label_gap
            self.draw_text(
                draw,
                (x, y),
                value,
                self.font_panel_value,
                (255, 94, 94, 255),
                stroke_fill=(70, 12, 18, 120),
            )
            y += value_height + row_gap

    def draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.ImageFont,
        fill: Tuple[int, int, int, int],
        stroke_fill: Tuple[int, int, int, int],
        stroke_width: int = 1,
    ) -> None:
        draw.text(
            position,
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )



def panel_rows(telemetry: Optional[TelemetryFrame]) -> List[Tuple[str, str]]:
    if telemetry is None:
        return [
            ("Version", "—"),
            ("Gear State", "—"),
            ("Frame Seq No", "—"),
            ("Vehicle Speed (m/s)", "—"),
            ("Accelerator Pedal Position", "—"),
            ("Steering Wheel Angle", "—"),
            ("Blinker On Left", "—"),
            ("Blinker On Right", "—"),
            ("Brake Applied", "—"),
            ("Autopilot State", "—"),
            ("Latitude", "—"),
            ("Longitude", "—"),
            ("Heading", "—"),
            ("Linear Accel X", "—"),
            ("Linear Accel Y", "—"),
            ("Linear Accel Z", "—"),
        ]
    return [
        ("Version", str(telemetry.version)),
        ("Gear State", telemetry.gear_label),
        ("Frame Seq No", str(telemetry.frame_seq_no)),
        ("Vehicle Speed (m/s)", f"{telemetry.vehicle_speed_mps:.2f}"),
        ("Accelerator Pedal Position", f"{telemetry.accelerator_pedal_position:.2f}"),
        ("Steering Wheel Angle", f"{telemetry.steering_wheel_angle:.2f}"),
        ("Blinker On Left", str(telemetry.blinker_on_left).lower()),
        ("Blinker On Right", str(telemetry.blinker_on_right).lower()),
        ("Brake Applied", str(telemetry.brake_applied).lower()),
        ("Autopilot State", telemetry.autopilot_label),
        ("Latitude", f"{telemetry.latitude_deg:.6f}"),
        ("Longitude", f"{telemetry.longitude_deg:.6f}"),
        ("Heading", f"{telemetry.heading_deg:.2f}"),
        ("Linear Accel X", f"{telemetry.linear_acceleration_mps2_x:.2f}"),
        ("Linear Accel Y", f"{telemetry.linear_acceleration_mps2_y:.2f}"),
        ("Linear Accel Z", f"{telemetry.linear_acceleration_mps2_z:.2f}"),
    ]


def measure_text(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return font.getsize(text)


def format_speed(speed_mps: float, unit: str) -> Tuple[str, str]:
    if unit == "kph":
        return str(int(round(speed_mps * 3.6))), "KPH"
    if unit == "mps":
        return f"{speed_mps:.1f}", "M/S"
    return str(int(round(speed_mps * 2.23693629))), "MPH"


def decode_frames_pyav(mp4_path: Path) -> Generator[Image.Image, None, None]:
    """
    Decode video using PyAV — a direct port of Tesla's per-frame VideoDecoder.

    Unlike the ffmpeg pipe, PyAV wraps libavcodec directly, giving frame-by-frame
    control with no VFR duplication, no timestamp rewriting, and no buffering
    that can desynchronize frame indices from the telemetry array.
    """
    container = _av.open(str(mp4_path))
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"
    try:
        for frame in container.decode(video_stream):
            yield Image.fromarray(frame.to_ndarray(format='rgb24'))
    finally:
        container.close()


def build_encoder(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    quality_mode: str,
    gpu_mode: str,
) -> Tuple[subprocess.Popen, str]:
    video_encoder, hardware_accelerated = choose_video_encoder(quality_mode, gpu_mode)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:a",
        "copy",
    ]

    if quality_mode == "lossless":
        command.extend(
            [
                "-c:v",
                "libx264rgb",
                "-preset",
                "veryslow",
                "-crf",
                "0",
                "-pix_fmt",
                "rgb24",
                "-movflags",
                "+faststart",
            ]
        )
    elif quality_mode == "compatible":
        if hardware_accelerated:
            command.extend(hardware_encoder_args(video_encoder, crf))
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    str(crf),
                    "-profile:v",
                    "high",
                    "-level:v",
                    "4.2",
                    "-movflags",
                    "+faststart",
                    "-tune",
                    "film",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )
    else:
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryslow",
                "-crf",
                str(crf),
                "-profile:v",
                "high444",
                "-level:v",
                "5.1",
                "-movflags",
                "+faststart",
                "-tune",
                "film",
                "-pix_fmt",
                "yuv444p",
            ]
        )

    command.append(str(output_path))
    return subprocess.Popen(command, stdin=subprocess.PIPE), video_encoder


def ffmpeg_supports_encoder(name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and name in result.stdout


def choose_video_encoder(quality_mode: str, gpu_mode: str) -> Tuple[str, bool]:
    if quality_mode != "compatible":
        return "libx264", False

    if gpu_mode == "off":
        return "libx264", False

    system = platform.system()
    candidates: List[str] = []

    if system == "Darwin":
        candidates = ["h264_videotoolbox"]
    elif system == "Windows":
        candidates = ["h264_nvenc", "h264_qsv", "h264_amf"]

    for encoder in candidates:
        if ffmpeg_supports_encoder(encoder):
            return encoder, True

    if gpu_mode == "on":
        print("Requested GPU acceleration, but no supported FFmpeg hardware encoder was found. Falling back to CPU.", file=sys.stderr)

    return "libx264", False


def hardware_encoder_args(encoder: str, crf: int) -> List[str]:
    if encoder == "h264_videotoolbox":
        bit_rate = max(12_000_000, 40_000_000 - (crf * 1_500_000))
        return [
            "-c:v",
            encoder,
            "-b:v",
            str(bit_rate),
            "-maxrate",
            str(int(bit_rate * 1.2)),
            "-bufsize",
            str(bit_rate * 2),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    if encoder == "h264_nvenc":
        return [
            "-c:v",
            encoder,
            "-preset",
            "p7",
            "-cq",
            str(min(18, max(10, crf + 6))),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    if encoder == "h264_qsv":
        return [
            "-c:v",
            encoder,
            "-global_quality",
            str(min(18, max(10, crf + 6))),
            "-look_ahead",
            "1",
            "-pix_fmt",
            "nv12",
            "-movflags",
            "+faststart",
        ]
    if encoder == "h264_amf":
        return [
            "-c:v",
            encoder,
            "-quality",
            "quality",
            "-rc",
            "cqp",
            "-qp_i",
            str(min(18, max(10, crf + 6))),
            "-qp_p",
            str(min(20, max(12, crf + 8))),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def default_image_pipeline_config() -> dict:
    return {
        "pipeline_order": [
            "black_white_point_remap",
            "gamma",
            "white_balance",
            "contrast",
            "saturation_vibrance",
            "dehaze_or_local_contrast",
            "sharpen",
            "light_denoise",
        ],
        "levels": {
            "black_point": 0.03,
            "white_point": 0.97,
        },
        "gamma": 1.02,
        "exposure_ev": 0.0,
        "white_balance": {
            "red_gain": 1.0,
            "green_gain": 1.0,
            "blue_gain": 1.0,
        },
        "contrast": 1.03,
        "saturation": 1.08,
        "vibrance": 0.08,
        "highlights": -0.03,
        "shadows": 0.04,
        "dehaze": 0.04,
        "clarity": 0.05,
        "sharpen": {
            "amount": 0.25,
            "radius": 1.0,
            "threshold": 0.03,
        },
        "denoise": 0.02,
        "vignette_correction": 0.0,
    }


def load_image_pipeline_config(source: Optional[str]) -> Optional[dict]:
    if source:
        text = source.strip()
        if not text:
            source = None

    if source:
        if text.startswith("{"):
            payload = text
        else:
            candidate = Path(text)
            if not candidate.exists():
                raise RuntimeError(f"Image params JSON file not found: {candidate}")
            payload = candidate.read_text(encoding="utf-8")
    else:
        candidate = DEFAULT_IMAGE_PARAMS_PATH
        if not candidate.exists():
            raise RuntimeError(f"Default image params JSON file not found: {candidate}")
        payload = candidate.read_text(encoding="utf-8")

    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise RuntimeError("Image params JSON must decode to an object.")
    return deep_merge(default_image_pipeline_config(), loaded)


def describe_image_profile_source(source: Optional[str]) -> str:
    if not source:
        return str(DEFAULT_IMAGE_PARAMS_PATH)
    text = source.strip()
    if text.startswith("{"):
        return "inline-json"
    return text


def create_color_mapping_lut(black_point: float, white_point: float, gamma_value: float) -> np.ndarray:
    """
    Combine levels and gamma adjustments into one LUT.

    The black/white points are normalized to [0.0, 1.0], then stretched to the
    0-255 range before applying gamma.
    """
    require_video_enhancement_dependencies()
    input_intensities = np.arange(256, dtype=np.float32)
    black = black_point * 255.0
    white = white_point * 255.0
    denominator = max(1.0, white - black)

    levels_map = (input_intensities - black) / denominator
    levels_map = np.clip(levels_map, 0.0, 1.0) * 255.0

    lut = np.uint8(
        np.clip((levels_map / 255.0) ** (1.0 / gamma_value), 0.0, 1.0) * 255.0
    )
    return lut


def enhance_saturation_bgr(frame_bgr: np.ndarray, factor: float) -> np.ndarray:
    require_video_enhancement_dependencies()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) * factor
    hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def enhance_saturation_bgr_umat(frame_bgr, factor: float):
    require_video_enhancement_dependencies()
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    s_channel = cv2.multiply(s_channel, factor)
    _, s_channel = cv2.threshold(s_channel, 255, 255, cv2.THRESH_TRUNC)
    hsv = cv2.merge((h_channel, s_channel, v_channel))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def choose_opencv_backend(accel_mode: str) -> str:
    require_video_enhancement_dependencies()

    has_opencl = bool(hasattr(cv2, "ocl") and cv2.ocl.haveOpenCL())
    if accel_mode == "off":
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
        return "cpu"

    if has_opencl:
        cv2.ocl.setUseOpenCL(True)
        return "opencl"

    if accel_mode == "on":
        raise RuntimeError(
            "OpenCV GPU acceleration was requested, but no OpenCL backend is available in this OpenCV build."
        )

    return "cpu"


def build_compiled_image_pipeline(config: dict) -> dict:
    require_video_enhancement_dependencies()
    compiled = deep_merge(default_image_pipeline_config(), config)
    compiled["_levels_lut"] = create_color_mapping_lut(
        float(compiled["levels"]["black_point"]),
        float(compiled["levels"]["white_point"]),
        1.0,
    )
    compiled["_gamma_lut"] = create_color_mapping_lut(
        0.0,
        1.0,
        max(0.01, float(compiled["gamma"])),
    )
    dehaze = float(compiled.get("dehaze", 0.0))
    compiled["_clahe"] = None
    if dehaze > 0.0:
        compiled["_clahe"] = cv2.createCLAHE(
            clipLimit=max(1.0, 1.0 + dehaze * 6.0),
            tileGridSize=(8, 8),
        )
    return compiled


def apply_levels_stage(bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(bgr, lut)


def apply_gamma_stage(bgr: np.ndarray, lut: np.ndarray, exposure_ev: float) -> np.ndarray:
    corrected = cv2.LUT(bgr, lut)
    if abs(exposure_ev) < 1e-6:
        return corrected
    exposure_scale = float(2.0 ** exposure_ev)
    return np.clip(corrected.astype(np.float32) * exposure_scale, 0, 255).astype(np.uint8)


def apply_white_balance_stage(bgr: np.ndarray, gains: dict) -> np.ndarray:
    working = bgr.astype(np.float32)
    working[:, :, 0] *= float(gains.get("blue_gain", 1.0))
    working[:, :, 1] *= float(gains.get("green_gain", 1.0))
    working[:, :, 2] *= float(gains.get("red_gain", 1.0))
    return np.clip(working, 0, 255).astype(np.uint8)


def apply_contrast_stage(
    bgr: np.ndarray,
    contrast: float,
    highlights: float,
    shadows: float,
) -> np.ndarray:
    working = bgr.astype(np.float32) / 255.0
    if abs(contrast - 1.0) > 1e-6:
        working = (working - 0.5) * contrast + 0.5

    luminance = np.max(working, axis=2, keepdims=True)
    if abs(highlights) > 1e-6:
        working = working + highlights * luminance * (1.0 - working)
    if abs(shadows) > 1e-6:
        shadow_mask = 1.0 - luminance
        working = working + shadows * shadow_mask * working

    return np.clip(working * 255.0, 0, 255).astype(np.uint8)


def apply_saturation_vibrance_stage(
    bgr: np.ndarray,
    saturation: float,
    vibrance: float,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    sat = sat * saturation
    if abs(vibrance) > 1e-6:
        sat = sat + vibrance * (1.0 - sat) * np.clip(1.0 - sat, 0.0, 1.0)
    hsv[:, :, 1] = np.clip(sat, 0.0, 1.0) * 255.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_dehaze_clarity_stage(
    bgr: np.ndarray,
    dehaze: float,
    clarity: float,
    clahe,
) -> np.ndarray:
    if dehaze <= 0.0 and clarity <= 0.0:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    if dehaze > 0.0 and clahe is not None:
        dehazed_l = clahe.apply(l_channel.astype(np.uint8)).astype(np.float32)
        blend = min(1.0, dehaze * 0.65)
        l_channel = cv2.addWeighted(l_channel, 1.0 - blend, dehazed_l, blend, 0.0)

    if clarity > 0.0:
        blurred_l = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=2.2)
        l_channel = cv2.addWeighted(
            l_channel,
            1.0 + clarity * 1.35,
            blurred_l,
            -clarity * 1.35,
            0.0,
        )

    lab[:, :, 0] = np.clip(l_channel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpen_stage(
    bgr: np.ndarray,
    amount: float,
    radius: float,
    threshold: float,
) -> np.ndarray:
    if amount <= 0.0:
        return bgr

    sigma = max(0.1, float(radius))
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    working = bgr.astype(np.float32)
    diff = working - blurred.astype(np.float32)
    sharpened = np.clip(working + diff * amount, 0, 255)

    if threshold > 0.0:
        mask = np.max(np.abs(diff), axis=2) >= (threshold * 255.0)
        result = working.copy()
        result[mask] = sharpened[mask]
        return result.astype(np.uint8)

    return sharpened.astype(np.uint8)


def apply_denoise_stage(bgr: np.ndarray, denoise: float) -> np.ndarray:
    if denoise <= 0.0:
        return bgr
    strength = max(1.0, denoise * 18.0)
    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=strength,
        hColor=max(1.0, strength * 0.8),
        templateWindowSize=7,
        searchWindowSize=21,
    )


def apply_vignette_correction_stage(bgr: np.ndarray, correction: float) -> np.ndarray:
    if abs(correction) < 1e-6:
        return bgr

    height, width = bgr.shape[:2]
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    center_x = width / 2.0
    center_y = height / 2.0
    radius = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    radius /= max(1.0, radius.max())

    mask = 1.0 + correction * (radius ** 2) * 0.35
    corrected = bgr.astype(np.float32) * mask[:, :, None]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def apply_video_enhancement(
    frame: Image.Image,
    compiled_config: dict,
    opencv_backend: str,
) -> Image.Image:
    require_video_enhancement_dependencies()
    rgb = np.array(frame)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if opencv_backend == "opencl" and hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(True)

    for stage in compiled_config.get("pipeline_order", []):
        if stage == "black_white_point_remap":
            bgr = apply_levels_stage(bgr, compiled_config["_levels_lut"])
        elif stage == "gamma":
            bgr = apply_gamma_stage(
                bgr,
                compiled_config["_gamma_lut"],
                float(compiled_config.get("exposure_ev", 0.0)),
            )
        elif stage == "white_balance":
            bgr = apply_white_balance_stage(bgr, compiled_config.get("white_balance", {}))
        elif stage == "contrast":
            bgr = apply_contrast_stage(
                bgr,
                float(compiled_config.get("contrast", 1.0)),
                float(compiled_config.get("highlights", 0.0)),
                float(compiled_config.get("shadows", 0.0)),
            )
        elif stage == "saturation_vibrance":
            bgr = apply_saturation_vibrance_stage(
                bgr,
                float(compiled_config.get("saturation", 1.0)),
                float(compiled_config.get("vibrance", 0.0)),
            )
        elif stage == "dehaze_or_local_contrast":
            bgr = apply_dehaze_clarity_stage(
                bgr,
                float(compiled_config.get("dehaze", 0.0)),
                float(compiled_config.get("clarity", 0.0)),
                compiled_config.get("_clahe"),
            )
        elif stage == "sharpen":
            sharpen = compiled_config.get("sharpen", {})
            bgr = apply_sharpen_stage(
                bgr,
                float(sharpen.get("amount", 0.0)),
                float(sharpen.get("radius", 1.2)),
                float(sharpen.get("threshold", 0.02)),
            )
        elif stage == "light_denoise":
            bgr = apply_denoise_stage(bgr, float(compiled_config.get("denoise", 0.0)))
        elif stage == "vignette_correction":
            bgr = apply_vignette_correction_stage(
                bgr,
                float(compiled_config.get("vignette_correction", 0.0)),
            )

    corrected_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(corrected_rgb)


def require_video_enhancement_dependencies() -> None:
    if cv2 is None or np is None:
        raise RuntimeError(
            "Video enhancement requires numpy and opencv-python-headless. "
            "Install the updated requirements.txt first."
        )


def render_video(
    input_path: Path,
    output_path: Path,
    panel_width: int,
    speed_unit: str,
    hud_position: str,
    hud_transparency: float,
    telemetry_offset_frames: int,
    telemetry_offset_seconds: float,
    enhance_video: bool,
    opencv_accel: str,
    image_params_json: Optional[str],
    crf: int,
    quality_mode: str,
    gpu_mode: str,
) -> None:
    video = probe_video(input_path)
    telemetry = read_frame_aligned_telemetry(input_path)
    if not telemetry:
        raise RuntimeError(
            "No frame-aligned Tesla SEI telemetry found in this MP4."
        )
    total_offset_frames = telemetry_offset_frames + int(round(telemetry_offset_seconds * video.fps))
    timeline = TelemetryTimeline(telemetry, video.frame_count, total_offset_frames)
    renderer = Renderer(
        video.width,
        video.height,
        panel_width,
        speed_unit,
        hud_position,
        hud_transparency,
    )
    image_adjustment_enabled = enhance_video or bool(image_params_json)
    compiled_image_pipeline = None
    opencv_backend = "disabled"
    image_profile_source = "disabled"
    if image_adjustment_enabled:
        compiled_image_pipeline = build_compiled_image_pipeline(
            load_image_pipeline_config(image_params_json) or default_image_pipeline_config()
        )
        opencv_backend = choose_opencv_backend(opencv_accel)
        image_profile_source = describe_image_profile_source(image_params_json)

    encoder, video_encoder = build_encoder(
        input_path=input_path,
        output_path=output_path,
        width=renderer.canvas_width,
        height=renderer.canvas_height,
        fps=video.fps,
        crf=crf,
        quality_mode=quality_mode,
        gpu_mode=gpu_mode,
    )

    print(
        "Runtime summary:\n"
        f"  decoder           = pyav\n"
        f"  encoder           = {video_encoder}\n"
        f"  quality_mode      = {quality_mode}\n"
        f"  opencv_backend    = {opencv_backend}\n"
        f"  image_profile     = {image_profile_source}\n"
        f"  telemetry_source  = tesla-parseframes\n"
        f"  telemetry_frames  = {len(timeline)}\n"
        f"  video_frames      = {video.frame_count}\n"
        f"  timeline_align    = {timeline.alignment_mode}\n"
        f"  telemetry_offset  = {timeline.offset_frames}\n"
        f"  hud_position      = {hud_position}\n"
        f"  hud_transparency  = {hud_transparency:.2f}",
        file=sys.stderr,
    )
    if timeline.frame_count_delta is not None and timeline.frame_count_delta != 0:
        print(
            "Frame count note: "
            f"telemetry_frames - video_frames = {timeline.frame_count_delta}. "
            "Telemetry remains on Tesla's original direct per-frame mapping; "
            "ffprobe frame totals are not used to stretch the overlay timeline.",
            file=sys.stderr,
        )

    assert encoder.stdin is not None

    total_frames = video.frame_count or 0
    frame_index = 0
    try:
        for frame in decode_frames_pyav(input_path):
            if image_adjustment_enabled and compiled_image_pipeline is not None:
                frame = apply_video_enhancement(
                    frame,
                    compiled_image_pipeline,
                    opencv_backend,
                )
            telemetry_frame = timeline.frame_at(frame_index)
            rendered = renderer.render(frame, telemetry_frame)
            encoder.stdin.write(rendered.tobytes())
            frame_index += 1
            if total_frames > 0:
                pct = frame_index * 100 // total_frames
                bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
                print(
                    f"\rRendering [{bar}] {pct}%  ({frame_index}/{total_frames})",
                    end="",
                    file=sys.stderr,
                )
            elif frame_index % 120 == 0:
                print(f"\rRendered {frame_index} frames...", end="", file=sys.stderr)
        if total_frames > 0:
            print(
                f"\rRendering [{'=' * 50}>] 100%  ({frame_index}/{frame_index})",
                file=sys.stderr,
            )
        else:
            print(f"\rRendered {frame_index} frames.", file=sys.stderr)
    finally:
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError("ffmpeg failed while encoding the overlay video.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Tesla dashcam MP4 with SEI telemetry panel and HUD overlay."
    )
    parser.add_argument("input_mp4", type=Path, help="Tesla dashcam MP4 file")
    parser.add_argument("output_mp4", type=Path, help="Path for the rendered MP4")
    parser.add_argument(
        "--panel-width",
        type=int,
        default=360,
        help="Width of the telemetry panel placed to the right of the video",
    )
    parser.add_argument(
        "--speed-unit",
        choices=("mph", "kph", "mps"),
        default="mph",
        help="Unit for the large in-video speed readout",
    )
    parser.add_argument(
        "--hud-position",
        choices=("top-left", "top-right", "bottom-left", "bottom-right"),
        default="top-right",
        help="Corner of the video where the compact HUD card is anchored.",
    )
    parser.add_argument(
        "--hud-transparency",
        type=float,
        default=0.54,
        help="HUD card transparency from 0.0 (opaque) to 1.0 (fully transparent).",
    )
    parser.add_argument(
        "--telemetry-offset-frames",
        type=int,
        default=0,
        help="Shift telemetry relative to the video by whole frames. Positive values pull telemetry earlier on screen; negative values delay it.",
    )
    parser.add_argument(
        "--telemetry-offset-seconds",
        type=float,
        default=0.0,
        help="Shift telemetry relative to the video by seconds. Positive values pull telemetry earlier on screen; negative values delay it.",
    )
    parser.add_argument(
        "--image-params-json",
        help="Path to a JSON file or an inline JSON object describing the image enhancement pipeline.",
    )
    parser.add_argument(
        "--enhance-video",
        action="store_true",
        help="Enable image enhancement using the bundled default JSON profile.",
    )
    parser.add_argument(
        "--opencv-accel",
        choices=("auto", "on", "off"),
        default="auto",
        help="OpenCV acceleration mode for video enhancement. Auto uses OpenCL/UMat when available.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=4,
        help="libx264 CRF value for output quality. Lower is higher quality.",
    )
    parser.add_argument(
        "--quality-mode",
        choices=("visually-lossless", "lossless", "compatible"),
        default="visually-lossless",
        help="Output encoding mode. 'visually-lossless' keeps 4:4:4 chroma, 'lossless' uses libx264rgb CRF 0, 'compatible' uses yuv420p.",
    )
    parser.add_argument(
        "--gpu",
        choices=("auto", "on", "off"),
        default="auto",
        help="Hardware encode selection. Auto-detects macOS/Windows GPU encoders for compatible mode only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input_mp4.exists():
        print(f"Input file not found: {args.input_mp4}", file=sys.stderr)
        return 1
    try:
        render_video(
            input_path=args.input_mp4,
            output_path=args.output_mp4,
            panel_width=args.panel_width,
            speed_unit=args.speed_unit,
            hud_position=args.hud_position,
            hud_transparency=args.hud_transparency,
            telemetry_offset_frames=args.telemetry_offset_frames,
            telemetry_offset_seconds=args.telemetry_offset_seconds,
            enhance_video=args.enhance_video,
            opencv_accel=args.opencv_accel,
            image_params_json=args.image_params_json,
            crf=args.crf,
            quality_mode=args.quality_mode,
            gpu_mode=args.gpu,
        )
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        return error.returncode or 1
    except Exception as error:  # pragma: no cover - CLI fallback
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
