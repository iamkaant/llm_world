from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from main import RunConfig, SugarscapeSimulation, save_outputs


class GuiCollector:
    def __init__(self, on_frame: Callable[[Dict[str, Any]], None], on_event: Callable[[Dict[str, Any]], None]) -> None:
        self._on_frame = on_frame
        self._on_event = on_event
        self._last_frame_index = 0

    def __enter__(self) -> "GuiCollector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def update(self, sim: SugarscapeSimulation) -> None:
        # Forward only new frames that were recorded by the simulation.
        while self._last_frame_index < len(sim.trace):
            frame = deepcopy(sim.trace[self._last_frame_index])
            self._last_frame_index += 1
            self._on_frame(frame)

    def add_event(self, event: Dict[str, Any]) -> None:
        self._on_event(deepcopy(event))

    def add_decision(self, agent_id: str, action: Any) -> None:
        # Decisions are already present inside frame["decision"], so no-op here.
        return


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def build_config_from_env() -> RunConfig:
    load_dotenv()

    backend = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    if backend not in {"openrouter", "ollama"}:
        backend = "ollama"

    if backend == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free").strip()
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_BACKEND=openrouter")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
    else:
        model = os.getenv("OLLAMA_MODEL", "gemma3:4b").strip()
        api_key = ""
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()

    return RunConfig(
        backend=backend,
        api_key=api_key,
        model=model,
        ollama_url=ollama_url,
        grid_size=_env_int("GRID_SIZE", 30),
        initial_agents=_env_int("INITIAL_AGENTS", 2),
        initial_energy=_env_int("INITIAL_ENERGY", 120),
        max_steps=_env_int("MAX_STEPS", 60),
        max_population=_env_int("MAX_POPULATION", 60),
        energy_spawn_probability=_env_float("ENERGY_SPAWN_PROBABILITY", 0.03),
        seed=_env_int("SEED", 42),
        scenario=os.getenv("SCENARIO", "baseline").strip(),
        max_requests_per_minute=_env_int("MAX_REQUESTS_PER_MINUTE", 2),
        min_seconds_between_requests=_env_float("MIN_SECONDS_BETWEEN_REQUESTS", 30.0),
        max_total_requests_per_run=_env_int("MAX_TOTAL_REQUESTS_PER_RUN", 400),
        max_total_tokens_per_run=_env_int("MAX_TOTAL_TOKENS_PER_RUN", 200000),
        temperature=_env_float("TEMPERATURE", 0.7),
        max_tokens=_env_int("MAX_TOKENS", 450),
        request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 30.0),
    )


class SimulationWorker(threading.Thread):
    def __init__(self, cfg: RunConfig, out_dir: Path, out_queue: "queue.Queue[tuple[str, Any]]") -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.out_dir = out_dir
        self.out_queue = out_queue

    def run(self) -> None:
        try:
            sim = SugarscapeSimulation(self.cfg)
            collector = GuiCollector(
                on_frame=lambda frame: self.out_queue.put(("frame", frame)),
                on_event=lambda event: self.out_queue.put(("event", event)),
            )
            summary = sim.run(collector)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            run_dir = self.out_dir / f"run-{timestamp}"
            save_outputs(summary, sim.events, sim.trace, run_dir)

            self.out_queue.put(
                (
                    "done",
                    {
                        "summary": summary,
                        "run_dir": str(run_dir),
                        "trace_path": str(run_dir / "trace.jsonl"),
                    },
                )
            )
        except Exception as exc:
            self.out_queue.put(("error", str(exc)))


class SimulationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LLM Sugarscape GUI")
        self.root.geometry("1400x900")

        self.event_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.worker: Optional[SimulationWorker] = None

        self.frames: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.current_index = -1
        self.is_playing = True
        self.simulation_running = False

        self.show_full_text = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready. Click Start Simulation or Load Trace.")

        self._build_ui()
        self.root.after(100, self._poll_queue)
        self.root.after(150, self._playback_tick)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Start Simulation", command=self.start_simulation).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load Trace", command=self.load_trace).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save Current Trace", command=self.save_current_trace).pack(side=tk.LEFT, padx=4)

        self.play_btn = ttk.Button(top, text="Pause", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="<< 10", command=lambda: self.step_relative(-10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="< 1", command=lambda: self.step_relative(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="1 >", command=lambda: self.step_relative(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="10 >>", command=lambda: self.step_relative(10)).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(
            top,
            text="Unfold Full Thoughts/Messages",
            variable=self.show_full_text,
            command=self._refresh_details,
        ).pack(side=tk.LEFT, padx=12)

        ttk.Label(top, textvariable=self.status_text).pack(side=tk.RIGHT)

        timeline_row = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        timeline_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(timeline_row, text="Timeline").pack(side=tk.LEFT)

        self.timeline = tk.Scale(
            timeline_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            showvalue=True,
            command=self._on_timeline_change,
            length=900,
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#121212", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._redraw_current_frame())

        right_top = ttk.LabelFrame(right, text="Frame Details", padding=8)
        right_top.pack(fill=tk.X, padx=4, pady=4)

        self.meta_text = tk.StringVar(value="No frame yet")
        ttk.Label(right_top, textvariable=self.meta_text, justify=tk.LEFT).pack(anchor=tk.W)

        right_mid = ttk.LabelFrame(right, text="Decisions (fold/unfold per step)", padding=8)
        right_mid.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.decision_tree = ttk.Treeview(right_mid, show="tree", height=14)
        self.decision_tree.pack(fill=tk.BOTH, expand=True)
        self.decision_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        right_bottom = ttk.LabelFrame(right, text="Decision Text", padding=8)
        right_bottom.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.detail_text = ScrolledText(right_bottom, wrap=tk.WORD, height=12)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        self.detail_text.configure(state=tk.DISABLED)

        events_frame = ttk.LabelFrame(right, text="Recent Events", padding=8)
        events_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.events_text = ScrolledText(events_frame, wrap=tk.WORD, height=9)
        self.events_text.pack(fill=tk.BOTH, expand=True)
        self.events_text.configure(state=tk.DISABLED)

    def start_simulation(self) -> None:
        if self.simulation_running:
            messagebox.showinfo("Simulation running", "A simulation is already running.")
            return

        try:
            cfg = build_config_from_env()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        self.frames.clear()
        self.events.clear()
        self.current_index = -1
        self.timeline.configure(to=0)
        self.timeline.set(0)
        self.decision_tree.delete(*self.decision_tree.get_children())
        self._set_detail_text("")
        self._set_events_text("")

        out_dir = Path("outputs")
        self.worker = SimulationWorker(cfg=cfg, out_dir=out_dir, out_queue=self.event_queue)
        self.worker.start()

        self.simulation_running = True
        self.is_playing = True
        self.play_btn.configure(text="Pause")
        self.status_text.set(f"Running scenario={cfg.scenario}, backend={cfg.backend}, model={cfg.model}")

    def load_trace(self) -> None:
        trace_path = filedialog.askopenfilename(
            title="Load trace.jsonl",
            initialdir=str(Path("outputs")),
            filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")],
        )
        if not trace_path:
            return

        frames: List[Dict[str, Any]] = []
        try:
            with Path(trace_path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    frames.append(json.loads(line))
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load trace: {exc}")
            return

        self.frames = frames
        self.events = []
        self.simulation_running = False
        self.current_index = 0 if self.frames else -1
        self.timeline.configure(to=max(0, len(self.frames) - 1))
        self.timeline.set(max(0, self.current_index))
        self.decision_tree.delete(*self.decision_tree.get_children())

        for idx, frame in enumerate(self.frames):
            self._add_decision_to_tree(idx, frame)

        if self.frames:
            self._render_frame(0)
        self.status_text.set(f"Loaded {len(self.frames)} frames from {trace_path}")

    def save_current_trace(self) -> None:
        if not self.frames:
            messagebox.showinfo("No frames", "No frames to save yet.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save trace.jsonl",
            defaultextension=".jsonl",
            filetypes=[("JSON Lines", "*.jsonl")],
            initialfile="trace.jsonl",
        )
        if not out_path:
            return

        try:
            with Path(out_path).open("w", encoding="utf-8") as f:
                for frame in self.frames:
                    f.write(json.dumps(frame) + "\n")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save trace: {exc}")
            return

        self.status_text.set(f"Saved {len(self.frames)} frames to {out_path}")

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        self.play_btn.configure(text="Pause" if self.is_playing else "Play")

    def step_relative(self, delta: int) -> None:
        if not self.frames:
            return
        target = max(0, min(len(self.frames) - 1, self.current_index + delta))
        self.timeline.set(target)
        self._render_frame(target)

    def _playback_tick(self) -> None:
        if self.is_playing and self.frames:
            if self.current_index < len(self.frames) - 1:
                next_index = self.current_index + 1
                self.timeline.set(next_index)
                self._render_frame(next_index)

        self.root.after(150, self._playback_tick)

    def _on_timeline_change(self, value: str) -> None:
        if not self.frames:
            return
        try:
            idx = int(float(value))
        except Exception:
            return
        self._render_frame(idx)

    def _poll_queue(self) -> None:
        drained = 0
        while True:
            try:
                item_type, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            drained += 1
            if item_type == "frame":
                self.frames.append(payload)
                self.timeline.configure(to=max(0, len(self.frames) - 1))
                self._add_decision_to_tree(len(self.frames) - 1, payload)

                # If user is at the latest frame, keep following live updates.
                if self.current_index in {-1, len(self.frames) - 2}:
                    self.current_index = len(self.frames) - 1
                    self.timeline.set(self.current_index)
                    self._render_frame(self.current_index)

            elif item_type == "event":
                self.events.append(payload)
                self._refresh_events()

            elif item_type == "done":
                self.simulation_running = False
                info = payload
                self.status_text.set(
                    f"Simulation complete. Frames={len(self.frames)}. Trace: {info.get('trace_path', '-') }"
                )

            elif item_type == "error":
                self.simulation_running = False
                self.status_text.set("Simulation failed. See error dialog.")
                messagebox.showerror("Simulation error", str(payload))

        if drained:
            self._refresh_details()

        self.root.after(100, self._poll_queue)

    def _refresh_events(self) -> None:
        recent = self.events[-120:]
        lines: List[str] = []
        for e in recent:
            step = e.get("step", "-")
            etype = e.get("type", "-")
            agent = e.get("agent", "-")
            if etype == "move":
                lines.append(f"step {step}: {agent} moved {e.get('from')} -> {e.get('to')} ({e.get('action')})")
            elif etype == "attack":
                lines.append(f"step {step}: {agent} attacked {e.get('target')}")
            elif etype == "share":
                lines.append(f"step {step}: {agent} shared {e.get('amount')} with {e.get('target')}")
            elif etype == "reproduce":
                lines.append(f"step {step}: {agent} reproduced -> {e.get('child')}")
            else:
                lines.append(f"step {step}: {etype} {agent}")

        self._set_events_text("\n".join(lines))

    def _add_decision_to_tree(self, idx: int, frame: Dict[str, Any]) -> None:
        d = frame.get("decision") or {}
        if not d:
            return

        title = f"frame {idx} | step {frame.get('step')} | {d.get('agent', '-')} -> {d.get('next_action', '-')}"
        node = self.decision_tree.insert("", tk.END, text=title, open=False)
        self.decision_tree.insert(node, tk.END, text=f"summary: {d.get('summary', '')}")
        self.decision_tree.insert(node, tk.END, text=f"thoughts: {d.get('thoughts', '')}")
        self.decision_tree.insert(node, tk.END, text=f"message: {d.get('message', '')}")

    def _on_tree_select(self, _event: Any) -> None:
        sel = self.decision_tree.selection()
        if not sel:
            return

        text = self.decision_tree.item(sel[0], "text")
        self._set_detail_text(text)

    def _render_frame(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.frames):
            return

        self.current_index = idx
        frame = self.frames[idx]
        self._draw_grid(frame)

        d = frame.get("decision") or {}
        m = frame.get("metrics") or {}
        self.meta_text.set(
            "\n".join(
                [
                    f"Frame: {idx + 1}/{len(self.frames)}",
                    f"Step: {frame.get('step')}  Phase: {frame.get('phase')}  Scenario: {frame.get('scenario')}",
                    f"Decision Agent: {d.get('agent', '-')}  Action: {d.get('next_action', '-')}",
                    f"Alive: {sum(1 for a in frame.get('agents', []) if a.get('alive'))}  Total agents seen: {len(frame.get('agents', []))}",
                    (
                        "Metrics: "
                        f"requests={m.get('requests', 0)}  tokens={m.get('total_tokens', 0)}  "
                        f"attacks={m.get('attacks', 0)}  shares={m.get('shares', 0)}  reproductions={m.get('reproductions', 0)}"
                    ),
                ]
            )
        )

        self._refresh_details()

    def _refresh_details(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.frames):
            return
        frame = self.frames[self.current_index]
        decision = frame.get("decision") or {}

        if not decision:
            self._set_detail_text("No agent decision in this frame.")
            return

        full = self.show_full_text.get()
        summary = str(decision.get("summary", ""))
        thoughts = str(decision.get("thoughts", ""))
        message = str(decision.get("message", ""))

        if not full:
            summary = self._truncate(summary, 220)
            thoughts = self._truncate(thoughts, 420)
            message = self._truncate(message, 420)

        content = (
            f"Agent: {decision.get('agent', '-') }\n"
            f"Action: {decision.get('next_action', '-') }\n"
            f"Share: target={decision.get('share_target')} amount={decision.get('share_amount')}\n"
            f"Attack: target={decision.get('attack_target')}\n\n"
            f"Summary:\n{summary}\n\n"
            f"Thoughts:\n{thoughts}\n\n"
            f"Message:\n{message}\n"
        )
        self._set_detail_text(content)

    def _draw_grid(self, frame: Dict[str, Any]) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())

        grid_size = int(frame.get("grid_size", 30))
        if grid_size <= 0:
            return

        cell = max(6, min(width // grid_size, height // grid_size))
        draw_w = cell * grid_size
        draw_h = cell * grid_size
        off_x = (width - draw_w) // 2
        off_y = (height - draw_h) // 2

        scenario = str(frame.get("scenario", "baseline"))
        poison_rows: set[int] = set()
        if scenario == "tradeoff":
            poison_rows = set(range(grid_size // 3, grid_size // 3 + 2))

        for y in range(grid_size):
            for x in range(grid_size):
                x0 = off_x + x * cell
                y0 = off_y + y * cell
                x1 = x0 + cell
                y1 = y0 + cell

                fill = "#1f1f1f"
                if scenario == "tradeoff" and y in poison_rows:
                    fill = "#4d1616"
                elif scenario == "tradeoff" and y <= 1:
                    fill = "#3a1f4f"

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#2c2c2c")

        for ec in frame.get("energy_cells", []):
            x = int(ec.get("x", 0))
            y = int(ec.get("y", 0))
            if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
                continue
            x0 = off_x + x * cell
            y0 = off_y + y * cell
            r = max(2, cell // 5)
            cx = x0 + cell // 2
            cy = y0 + cell // 2
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#d8b11e", outline="")

        alive_agents: Dict[tuple[int, int], List[str]] = {}
        for a in frame.get("agents", []):
            if not bool(a.get("alive", False)):
                continue
            pos = (int(a.get("x", 0)), int(a.get("y", 0)))
            alive_agents.setdefault(pos, []).append(str(a.get("id", "?")))

        for (x, y), ids in alive_agents.items():
            if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
                continue
            x0 = off_x + x * cell
            y0 = off_y + y * cell
            pad = max(1, cell // 10)
            self.canvas.create_oval(
                x0 + pad,
                y0 + pad,
                x0 + cell - pad,
                y0 + cell - pad,
                fill="#2fc96c",
                outline="#0f6f34",
                width=1,
            )

            label = ids[0]
            if len(ids) > 1:
                label = f"{ids[0]}+{len(ids)-1}"
            label = label.replace("Agent", "A")
            self.canvas.create_text(
                x0 + cell // 2,
                y0 + cell // 2,
                text=label,
                fill="#0a2212",
                font=("Helvetica", max(7, cell // 4), "bold"),
            )

    def _redraw_current_frame(self) -> None:
        if self.current_index >= 0 and self.current_index < len(self.frames):
            self._draw_grid(self.frames[self.current_index])

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _set_detail_text(self, content: str) -> None:
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, content)
        self.detail_text.configure(state=tk.DISABLED)

    def _set_events_text(self, content: str) -> None:
        self.events_text.configure(state=tk.NORMAL)
        self.events_text.delete("1.0", tk.END)
        self.events_text.insert(tk.END, content)
        self.events_text.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
