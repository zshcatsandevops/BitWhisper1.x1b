#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BitWhisker0.910b — a lightweight, desktop chat UI inspired by modern chat sites.
This is a fresh, legally clean design (not a pixel-for-pixel clone) that runs offline.
It includes:
  • Dark/Light modes
  • Streaming-style assistant typing
  • New Chat / multi-thread sidebar
  • Copy message
  • Save transcript
  • Integration with OpenAI's GPT-OSS-20B model (requires installation of dependencies and sufficient hardware)

Python 3.9+ recommended. Requires: tkinter (usually bundled), torch, transformers, accelerate.

Run:
  python bitwhisker0_910b.py
"""

import os
import sys
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

APP_NAME = "BitWhisker0.910b"


# =====================================================
# GPT-OSS-20B model integration
# =====================================================

class BitWhiskerModel:
    """
    Integration with OpenAI's GPT-OSS-20B model using Hugging Face Transformers.
    This replaces the toy model with a real LLM for text generation.
    Supports streaming and temperature control.
    """
    def __init__(self, temperature: float = 0.9, seed: Optional[int] = None):
        self.temperature = max(0.05, min(float(temperature), 1.5))
        model_id = "openai/gpt-oss-20b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use Accelerate for efficient loading
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        self.model = load_checkpoint_and_dispatch(self.model, model_id, device_map="auto", dtype=torch.float16)
        # Seed is ignored for this model

    def generate(self, messages: List[dict], max_tokens: int = 120, stop_event: Optional[Callable[[], bool]] = None):
        """
        Stream-like generator yielding text chunks.
        Uses the chat template for formatting messages.
        Note: Stopping mid-generation halts UI updates but may not immediately stop the model computation.
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            "inputs": inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 50,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if stop_event and stop_event():
                break
            yield new_text

        thread.join()


# =====================================================
# Chat model/controller
# =====================================================

@dataclass
class Message:
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    id: int = field(default_factory=lambda: int(time.time() * 1e6) & 0x7FFFFFFF)


@dataclass
class ChatThread:
    title: str
    messages: List[Message] = field(default_factory=list)

    def to_dict(self):
        return {"title": self.title, "messages": [m.__dict__ for m in self.messages]}


class ChatController:
    def __init__(self):
        self.threads: List[ChatThread] = [ChatThread(title="New Chat")]
        self.current_index = 0
        self.temperature = 0.9
        self.max_tokens = 160
        self.model = BitWhiskerModel(temperature=self.temperature)

    @property
    def current(self) -> ChatThread:
        return self.threads[self.current_index]

    def new_thread(self, title="New Chat"):
        self.threads.insert(0, ChatThread(title=title))
        self.current_index = 0

    def set_temperature(self, t: float):
        self.temperature = max(0.05, min(float(t), 1.5))
        self.model = BitWhiskerModel(temperature=self.temperature)

    def set_max_tokens(self, n: int):
        self.max_tokens = max(1, min(int(n), 1000))


# =====================================================
# UI
# =====================================================

class BitWhiskerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} — Desktop")
        self.geometry("1100x720")
        self.minsize(920, 600)

        # Theming palettes
        self.theme = tk.StringVar(value="dark")  # 'dark' | 'light'
        self.palette = {
            "dark": {
                "bg": "#0f1115",
                "bg2": "#151821",
                "panel": "#10131a",
                "text": "#e8e8e8",
                "muted": "#9aa2b1",
                "accent": "#4f8cff",
                "bubble_user": "#1e2533",
                "bubble_ai": "#151b26",
                "border": "#1d2230"
            },
            "light": {
                "bg": "#f5f6f8",
                "bg2": "#ffffff",
                "panel": "#ffffff",
                "text": "#1f2430",
                "muted": "#5b6372",
                "accent": "#2e6bd3",
                "bubble_user": "#e9eef9",
                "bubble_ai": "#eef1f6",
                "border": "#dcdfe6"
            }
        }

        self.controller = ChatController()
        self._streaming = False
        self._stop_flag = False
        self._after_token = None

        self._build_style()
        self._build_layout()
        self._apply_theme()

    # --------------------------
    # Style
    # --------------------------
    def _build_style(self):
        self.style = ttk.Style(self)
        # Use 'clam' to allow custom color maps
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure("Sidebar.TFrame", background="#151821")
        self.style.configure("Header.TFrame", background="#0f1115")
        self.style.configure("Footer.TFrame", background="#0f1115")
        self.style.configure("Primary.TButton", padding=8)
        self.style.configure("Flat.TButton", padding=6)
        self.style.configure("TCheckbutton", background="#0f1115")

    # --------------------------
    # Layout
    # --------------------------
    def _build_layout(self):
        pal = self._p()

        # Header
        self.header = ttk.Frame(self, style="Header.TFrame")
        self.header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._title_label = tk.Label(
            self.header, text=APP_NAME, font=("Inter", 14, "bold"),
            bd=0, padx=12, pady=10
        )
        self._title_label.pack(side=tk.LEFT)

        self._mode_btn = ttk.Button(self.header, text="Toggle Theme", command=self._toggle_theme, style="Flat.TButton")
        self._mode_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        self._settings_btn = ttk.Button(self.header, text="Settings", command=self._open_settings, style="Flat.TButton")
        self._settings_btn.pack(side=tk.RIGHT, padx=6, pady=6)

        # Sidebar
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame")
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0, minsize=260)

        self._new_btn = ttk.Button(self.sidebar, text="＋  New Chat", command=self._new_chat, style="Primary.TButton")
        self._new_btn.pack(fill="x", padx=10, pady=(10, 6))

        self.thread_list = tk.Listbox(self.sidebar, activestyle="dotbox", highlightthickness=0, exportselection=False)
        self.thread_list.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.thread_list.bind("<<ListboxSelect>>", self._select_thread)
        self._refresh_thread_list()

        # Main area (messages + input)
        self.main = tk.Frame(self, bd=0, highlightthickness=0)
        self.main.grid(row=1, column=1, sticky="nsew")

        # Scrollable message area
        self.canvas = tk.Canvas(self.main, bd=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.msg_area = tk.Frame(self.canvas)
        self.msg_area.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._msg_window = self.canvas.create_window((0,0), window=self.msg_area, anchor="nw")

        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self._msg_window, width=e.width))

        # Footer input area
        self.footer = ttk.Frame(self, style="Footer.TFrame")
        self.footer.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.grid_rowconfigure(2, weight=0)

        self.input_box = tk.Text(self.footer, height=3, wrap="word", bd=0, padx=12, pady=10)
        self.input_box.grid(row=0, column=0, sticky="nsew", padx=(10,6), pady=10)
        self.input_box.bind("<Return>", self._enter_send)
        self.input_box.bind("<Shift-Return>", lambda e: None)  # allow newline with Shift+Enter

        self.btn_send = ttk.Button(self.footer, text="Send →", command=self._on_send, style="Primary.TButton")
        self.btn_send.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)

        self.btn_stop = ttk.Button(self.footer, text="Stop", command=self._stop_stream, style="Flat.TButton")
        self.btn_stop.grid(row=0, column=2, sticky="nsew", padx=(0,10), pady=10)

        self.footer.grid_columnconfigure(0, weight=1)

        # Keyboard focus
        self.input_box.focus_set()

        # Greeting
        self._add_message("assistant", f"{APP_NAME} is online. Ask me anything!")

        # Apply theme colors now that widgets exist
        self._apply_theme()

    # --------------------------
    # Helpers
    # --------------------------
    def _p(self):
        return self.palette[self.theme.get()]

    def _apply_theme(self):
        pal = self._p()
        # Window backgrounds
        self.configure(bg=pal["bg"])
        for fr in (self.main, self.msg_area):
            fr.configure(bg=pal["bg"])

        # Canvas
        self.canvas.configure(bg=pal["bg"])

        # Header/labels
        self._title_label.configure(bg=pal["bg"], fg=pal["text"])

        # Sidebar
        self.thread_list.configure(
            bg=pal["panel"], fg=pal["text"], selectbackground=pal["accent"],
            selectforeground="#ffffff", bd=0, highlightthickness=0
        )
        # Footer and input
        self.input_box.configure(
            bg=pal["bg2"], fg=pal["text"], insertbackground=pal["text"], highlightthickness=1,
            highlightbackground=pal["border"], highlightcolor=pal["accent"]
        )
        # Buttons (ttk picks system style; acceptable)

        # Repaint message bubbles
        for child in self.msg_area.winfo_children():
            if hasattr(child, "_bubble_role"):
                role = child._bubble_role
                bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
                for c in child.winfo_children():
                    if isinstance(c, tk.Label):
                        c.configure(bg=bg, fg=pal["text"])
                child.configure(bg=pal["bg"])

        # Header/Footer backgrounds via style updates
        self.style.configure("Header.TFrame", background=pal["bg"])
        self.style.configure("Footer.TFrame", background=pal["bg"])
        self.style.configure("Sidebar.TFrame", background=pal["panel"])

        self.update_idletasks()

    def _toggle_theme(self):
        self.theme.set("light" if self.theme.get() == "dark" else "dark")
        self._apply_theme()

    def _new_chat(self):
        self.controller.new_thread(title="New Chat")
        self._refresh_thread_list()
        self._clear_messages()
        self._add_message("assistant", f"Started a new conversation in {APP_NAME}.")

    def _refresh_thread_list(self):
        self.thread_list.delete(0, tk.END)
        for i, th in enumerate(self.controller.threads):
            title = th.title if th.title.strip() else "Untitled"
            if len(title) > 36:
                title = title[:33] + "..."
            self.thread_list.insert(tk.END, title)
        self.thread_list.selection_clear(0, tk.END)
        self.thread_list.selection_set(self.controller.current_index)
        self.thread_list.activate(self.controller.current_index)

    def _select_thread(self, event=None):
        sel = self.thread_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.controller.current_index = idx
        self._render_thread()

    def _render_thread(self):
        self._clear_messages()
        for msg in self.controller.current.messages:
            self._add_message(msg.role, msg.content, remember=False)
        self._scroll_to_bottom()

    def _clear_messages(self):
        for child in self.msg_area.winfo_children():
            child.destroy()

    def _add_message(self, role: str, content: str, remember: bool = True):
        pal = self._p()
        container = tk.Frame(self.msg_area, bg=pal["bg"], padx=12, pady=8)
        container._bubble_role = role  # mark for theme updates
        container.pack(fill="x", anchor="w")

        # Role label (muted)
        role_name = "You" if role == "user" else "Assistant"
        lbl_role = tk.Label(container, text=role_name, font=("Inter", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        # Bubble
        bubble_bg = pal["bubble_user"] if role == "user" else pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt = tk.Label(bubble, text=content, justify="left", anchor="w",
                       font=("Inter", 11), wraplength=820, bg=bubble_bg, fg=pal["text"])
        txt.pack(anchor="w", fill="x")

        # Tools row (copy)
        tools = tk.Frame(container, bg=pal["bg"])
        tools.pack(anchor="w", padx=6, pady=(0,6))
        btn_copy = ttk.Button(tools, text="Copy", command=lambda c=content: self._copy_text(c), style="Flat.TButton")
        btn_copy.pack(side="left")

        if remember:
            self.controller.current.messages.append(Message(role=role, content=content))

        self.update_idletasks()
        self._scroll_to_bottom()

    def _copy_text(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # now it stays on the clipboard
        messagebox.showinfo("Copied", "Message copied to clipboard.")

    def _scroll_to_bottom(self):
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _enter_send(self, event):
        if event.state & 0x1:  # Shift pressed -> allow newline
            return
        self._on_send()
        return "break"

    def _on_send(self):
        if self._streaming:
            return
        text = self.input_box.get("1.0", tk.END).strip()
        if not text:
            return
        self.input_box.delete("1.0", tk.END)

        # Update thread title on first user message
        if not self.controller.current.messages:
            self.controller.current.title = text.splitlines()[0][:60]
            self._refresh_thread_list()

        self._add_message("user", text)

        # Start assistant streaming response
        self._stream_assistant()

    def _stop_stream(self):
        self._stop_flag = True

    def _stream_assistant(self):
        self._stop_flag = False
        self._streaming = True
        pal = self._p()

        # Placeholder assistant bubble to fill progressively
        container = tk.Frame(self.msg_area, bg=pal["bg"], padx=12, pady=8)
        container._bubble_role = "assistant"
        container.pack(fill="x", anchor="w")

        lbl_role = tk.Label(container, text="Assistant", font=("Inter", 9, "bold"),
                            fg=pal["muted"], bg=pal["bg"])
        lbl_role.pack(anchor="w", padx=6)

        bubble_bg = pal["bubble_ai"]
        bubble = tk.Frame(container, bg=bubble_bg, bd=0, padx=12, pady=10, highlightthickness=1)
        bubble.configure(highlightbackground=pal["border"])
        bubble.pack(anchor="w", padx=6, pady=(2, 4), fill="x")

        txt_var = tk.StringVar(value="")
        txt = tk.Label(bubble, textvariable=txt_var, justify="left", anchor="w",
                       font=("Inter", 11), wraplength=820, bg=bubble_bg, fg=pal["text"])
        txt.pack(anchor="w", fill="x")

        tools = tk.Frame(container, bg=pal["bg"])
        tools.pack(anchor="w", padx=6, pady=(0,6))
        btn_copy = ttk.Button(tools, text="Copy", command=lambda: self._copy_text(txt_var.get()), style="Flat.TButton")
        btn_copy.pack(side="left")

        # Build messages list from history
        messages = [{"role": m.role, "content": m.content} for m in self.controller.current.messages]

        # Generation loop using .after to avoid blocking UI thread
        gen = self.controller.model.generate(messages, max_tokens=self.controller.max_tokens,
                                             stop_event=lambda: self._stop_flag)

        def step():
            nonlocal gen
            try:
                # Emit chunks as they come for a streamy feel
                chunk = next(gen)
                txt_var.set(txt_var.get() + chunk)
                self._scroll_to_bottom()
                self._after_token = self.after(10, step)
            except StopIteration:
                # Done
                self._finalize_assistant_message(txt_var.get())
            except Exception as e:
                self._finalize_assistant_message(f"[error] {e!r}")

            if self._stop_flag:
                self._finalize_assistant_message(txt_var.get())

        step()

    def _finalize_assistant_message(self, content: str):
        self._streaming = False
        self._stop_flag = False
        if self._after_token:
            try:
                self.after_cancel(self._after_token)
            except Exception:
                pass
            self._after_token = None
        # Record message
        self.controller.current.messages.append(Message(role="assistant", content=content))

    # --------------------------
    # Settings / Save
    # --------------------------
    def _open_settings(self):
        pal = self._p()
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("420x260")
        win.configure(bg=pal["bg"])

        tk.Label(win, text="Settings", font=("Inter", 12, "bold"), bg=pal["bg"], fg=pal["text"]).pack(pady=(10, 6))

        frm = tk.Frame(win, bg=pal["bg"])
        frm.pack(fill="x", expand=True, padx=16, pady=6)

        # Temperature
        tk.Label(frm, text="Temperature", bg=pal["bg"], fg=pal["text"]).grid(row=0, column=0, sticky="w", pady=6)
        t_var = tk.DoubleVar(value=self.controller.temperature)
        t_scale = tk.Scale(frm, from_=0.1, to=1.5, resolution=0.05, orient="horizontal",
                           variable=t_var, bg=pal["bg"], fg=pal["text"], highlightthickness=0)
        t_scale.grid(row=0, column=1, sticky="ew", padx=8)

        # Max tokens
        tk.Label(frm, text="Max tokens", bg=pal["bg"], fg=pal["text"]).grid(row=1, column=0, sticky="w", pady=6)
        n_var = tk.IntVar(value=self.controller.max_tokens)
        n_scale = tk.Scale(frm, from_=16, to=1000, resolution=1, orient="horizontal",
                           variable=n_var, bg=pal["bg"], fg=pal["text"], highlightthickness=0)
        n_scale.grid(row=1, column=1, sticky="ew", padx=8)

        frm.grid_columnconfigure(1, weight=1)

        # Action buttons
        bar = tk.Frame(win, bg=pal["bg"])
        bar.pack(fill="x", padx=16, pady=10)
        ttk.Button(bar, text="Save", command=lambda: self._apply_settings(t_var.get(), n_var.get())).pack(side="right")
        ttk.Button(bar, text="Save Transcript…", command=self._save_transcript).pack(side="left")

        win.transient(self)
        win.grab_set()
        win.focus_set()

    def _apply_settings(self, temp: float, max_tokens: int):
        self.controller.set_temperature(temp)
        self.controller.set_max_tokens(max_tokens)
        messagebox.showinfo("Settings", "Updated generation settings.")

    def _save_transcript(self):
        th = self.controller.current
        data = th.to_dict()
        default_name = f"{APP_NAME.replace(' ', '_').lower()}_{int(time.time())}.json"
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_name
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Transcript saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")

# =====================================================
# Main
# =====================================================

def main():
    app = BitWhiskerApp()
    app.mainloop()

if __name__ == "__main__":
    main()
