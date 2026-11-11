"""Agent mode (spec 5.2): the model calls the tools itself in a loop."""
from __future__ import annotations

from core.labeler.context import LabelContext
from core.labeler.conversation import CostGuard, call_model, repair_loop
from core.labeler.finalize import RawResult
from core.labeler.parse_reply import AgentReply, ReplyParseError, parse_agent_reply
from core.labeler.tool_dispatch import TOOL_SPECS, run_tool
from core.labeler.transcript import Transcript
from core.models import Question
from core.providers.base import ImagePart, Message, VisionProvider
from core.tools.crop import encode_png

MAX_TOOL_ERRORS = 4


class AgentModeFailed(RuntimeError):
    """The model could not use the tools properly: fall back to pipeline mode."""


def _start(ctx: LabelContext) -> Message:
    text = (f"Label this image. Camera: {'grayscale' if ctx.image.is_grayscale else 'color'}. "
            f"Image size shown: {ctx.shown.shape[1]}x{ctx.shown.shape[0]} px. "
            "Start with detect_laser, then follow the instructions.")
    if ctx.hint_points:
        text += (" The reviewer pointed at these places in the original image, each a point on a laser, not its box: "
                 + "; ".join(f"({x:.0f}, {y:.0f})" for x, y in ctx.hint_points)
                 + ". Check each one (detect_laser includes them); decide from the picture what is there.")
    if ctx.user_note:
        text += f" Note from the reviewer about this image (follow it): {ctx.user_note}"
    return Message("user", text, images=[ImagePart(encode_png(ctx.shown))])


def _answer_tools(ctx: LabelContext, reply, messages, transcript) -> int:
    errors = 0
    for call in reply.tool_calls:
        result = run_tool(ctx, call)
        errors += result.is_error
        message = Message("tool", result.text, images=result.images, tool_call_id=call.id, tool_name=call.name)
        messages.append(message)
        transcript.add_message(message)
    return errors


def _to_raw(ctx: LabelContext, parsed: AgentReply) -> RawResult:
    raw = RawResult(questions=list(parsed.questions) + list(ctx.questions))
    ctx.read_model_boxes([i["box"] for i in parsed.laser_boxes], [i["box"] for i in parsed.object_boxes])
    for item in parsed.laser_boxes:
        raw.laser_boxes.append({**item, "box": ctx.from_model(item["box"])})
    for item in parsed.object_boxes:
        raw.object_boxes.append({**item, "box": ctx.from_model(item["box"])})
    if ctx.coordinates_used != "pixels":
        raw.notes = (f"Box numbers were read as {ctx.coordinates_used} and converted to the picture's pixels"
                     + (f" ({ctx.coordinates_why})." if ctx.coordinates_why else "."))
    return raw


def run_agent(ctx: LabelContext, provider: VisionProvider, system_prompt: str, transcript: Transcript, *,
              max_steps: int = 12, max_repairs: int = 2, guard: CostGuard | None = None) -> RawResult:
    if not provider.supports_tools:
        raise AgentModeFailed("this provider does not support tool calls")
    messages = [Message("system", system_prompt), _start(ctx)]
    for message in messages:
        transcript.add_message(message)
    errors, used_tools = 0, False
    for _ in range(max_steps):
        reply = call_model(provider, messages, transcript, TOOL_SPECS, guard)
        if reply.tool_calls:
            used_tools = True
            errors += _answer_tools(ctx, reply, messages, transcript)
            if errors >= MAX_TOOL_ERRORS:
                raise AgentModeFailed(f"{errors} tool calls were invalid")
            continue
        if not used_tools:
            raise AgentModeFailed("the model answered without calling any tool")
        try:
            parsed, repairs = repair_loop(provider, messages, transcript, reply, parse_agent_reply, max_repairs, guard)
        except ReplyParseError as exc:
            raise AgentModeFailed(f"final answer was not valid after repairs: {exc}") from exc
        raw = _to_raw(ctx, parsed)
        if repairs:
            raw.notes = (raw.notes + f" final reply needed {repairs} repair(s).").strip()
        return raw
    raise AgentModeFailed(f"no final answer after {max_steps} steps")
