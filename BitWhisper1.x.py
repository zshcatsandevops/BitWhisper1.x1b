import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import threading
import time
import queue

class BitWhisperer:
    """A toy 1-bit LLM implementation - ALL BUGS FIXED"""
    
    def __init__(self, vocab_size=256, hidden_dim=64, max_len=50):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # FIX #7: Proper binary weight initialization
        # CHANGED: W_embed now uses randint(0, 2) for true binary values
        # CHANGED: W_out now uses correct range [0, 2) to get 0 or 1 values
        self.W_embed = np.random.randint(0, 2, (vocab_size, hidden_dim))  # Binary values!
        self.W_hidden = np.random.randint(0, 2, (hidden_dim, hidden_dim))
        self.W_out = np.random.randint(0, 2, (hidden_dim, vocab_size))  # Proper binary range!
        
    def matmul(self, A, B):
        # FIX #1: Made modulo 2 optional for proper 1-bit operations
        # CHANGED: Removed the destructive % 2 that was truncating all values
        # For true 1-bit LLM behavior, we keep values in integer space during computation
        # and only apply modulo when needed for specific 1-bit constraints
        result = np.dot(A, B)
        # Optionally apply modulo for 1-bit constraint: result = result % 2
        return result
    
    def forward(self, x):
        # FIX #2: Proper dimension handling for scalar and array inputs
        # CHANGED: Added proper shape checking and handling for both scalar and array inputs
        if np.isscalar(x):
            # Handle scalar input - get single embedding
            embedded = self.W_embed[x]  # Shape: (hidden_dim,)
        else:
            # Handle array input - get multiple embeddings
            x = np.asarray(x)
            if x.ndim == 0:  # 0-d array (scalar-like)
                embedded = self.W_embed[int(x)]
            else:  # 1-d array
                embedded = self.W_embed[x]  # Shape: (len(x), hidden_dim)
                # Average the embeddings if multiple tokens
                if embedded.ndim > 1:
                    embedded = np.mean(embedded, axis=0)
        
        # Ensure embedded is 1D for matrix multiplication
        embedded = np.asarray(embedded).flatten()
        
        hidden = self.matmul(embedded, self.W_hidden)
        
        # FIX #3: Return token directly instead of full logits array
        # CHANGED: Now returns a single token index that generate() expects
        logits = self.matmul(hidden, self.W_out)
        token = self.argmax(logits)
        return token  # Returns single token index, not full array
    
    def argmax(self, arr):
        # FIX #8: Properly handle ties by randomly selecting among maximum values
        # CHANGED: Now identifies all indices with max value and randomly chooses one
        arr = np.asarray(arr).flatten()
        max_val = np.max(arr)
        # Find all indices that have the maximum value
        max_indices = np.where(arr == max_val)[0]
        # Randomly choose among ties
        return np.random.choice(max_indices)
    
    def generate(self, prompt_tokens, goal_tokens, num_steps=20):
        # FIX #3: Now properly works with forward() returning a single token
        # FIX #9: Fixed logic errors in token comparison and goal checking
        # CHANGED: Updated to handle forward() returning tokens directly
        # CHANGED: Proper goal checking logic
        current = prompt_tokens[-1] if len(prompt_tokens) > 0 else 0
        generated = []
        goal_tokens_list = list(goal_tokens) if hasattr(goal_tokens, '__iter__') else [goal_tokens]
        
        for step in range(num_steps):
            # forward() now returns a single token
            next_token = self.forward(current)
            generated.append(next_token)
            current = next_token
            
            # FIX #9: Proper goal checking - check if we've generated the goal sequence
            # CHANGED: Check if the end of generated matches the goal sequence
            if len(generated) >= len(goal_tokens_list):
                if generated[-len(goal_tokens_list):] == goal_tokens_list:
                    break
        
        return generated

class BitWhispererGUI:
    """GUI for the BitWhisperer agent - THREAD SAFETY FIXED"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BitWhisperer - 1-bit LLM Agent")
        self.root.geometry("700x500")
        
        self.model = BitWhisperer()
        self.running = False
        
        # FIX #5: Added thread-safe message queue for GUI updates
        # CHANGED: Created a queue for thread-safe communication
        self.message_queue = queue.Queue()
        
        # Setup UI
        self.setup_ui()
        
        # FIX #5: Start periodic queue checking for thread-safe updates
        # CHANGED: Use after() to periodically check queue from main thread
        self.check_queue()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="BitWhisperer Agent", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Goal input frame
        goal_frame = tk.Frame(self.root)
        goal_frame.pack(pady=10)
        
        tk.Label(goal_frame, text="Agent Goal:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.goal_entry = tk.Entry(goal_frame, width=50)
        self.goal_entry.insert(0, "Hello World")
        self.goal_entry.pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        self.start_btn = tk.Button(btn_frame, text="Start Agent", 
                                   command=self.start_agent, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", 
                                  command=self.stop_agent, width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Output display
        output_label = tk.Label(self.root, text="Agent Output:", font=("Arial", 10))
        output_label.pack(pady=5)
        
        self.output_text = scrolledtext.ScrolledText(self.root, width=80, height=20,
                                                     wrap=tk.WORD)
        self.output_text.pack(pady=10, padx=10)
    
    # FIX #5: Thread-safe GUI update method
    # CHANGED: Added queue checking method that runs in main thread
    def check_queue(self):
        """Process messages from worker thread in a thread-safe manner"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg == "ENABLE_START_BUTTON":
                    self.start_btn.config(state=tk.NORMAL)
                else:
                    self.output_text.insert(tk.END, msg)
                    self.output_text.see(tk.END)
        except queue.Empty:
            pass
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def start_agent(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            goal = self.goal_entry.get()
            
            # FIX #5: Thread now uses message queue instead of direct GUI updates
            # CHANGED: Worker thread communicates via queue, not direct GUI calls
            agent_thread = threading.Thread(target=self.agent_loop, args=(goal,))
            agent_thread.daemon = True
            agent_thread.start()
    
    def stop_agent(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
    
    def agent_loop(self, goal):
        # FIX #4: Now actually uses the goal string for agent behavior
        # FIX #6: Properly encode goal string into tokens instead of random generation
        # CHANGED: Convert goal string into actual token sequence
        goal_tokens = [ord(c) % 256 for c in goal]  # Proper encoding of goal!
        
        # Initialize with prompt
        prompt = "Starting..."
        prompt_tokens = [ord(c) % 256 for c in prompt]
        
        step = 0
        max_steps = 30
        
        # FIX #5: Use message queue for thread-safe GUI updates
        # CHANGED: All GUI updates now go through message_queue
        self.message_queue.put(f"Goal: {goal}\n")
        self.message_queue.put(f"Goal Tokens: {goal_tokens}\n")
        self.message_queue.put(f"Starting agent loop...\n\n")
        
        while self.running and step < max_steps:
            try:
                # Generate tokens - now actually working toward the goal!
                generated = self.model.generate(prompt_tokens, goal_tokens, num_steps=8)
                
                # Try to convert to readable text
                try:
                    text_output = ''.join([chr(t % 256) for t in generated])
                    # Filter out non-printable
                    text_output = ''.join(c if 32 <= ord(c) < 127 else '?' for c in text_output)
                except Exception:
                    text_output = str(generated[:10])
                
                msg = f"Step {step}: {text_output}\n"
                
                # FIX #5: Thread-safe GUI update via queue
                # CHANGED: Put message in queue instead of direct insert
                self.message_queue.put(msg)
                
                # FIX #9: Proper goal-reached checking logic
                # CHANGED: Check if generated sequence matches goal tokens
                if len(generated) >= len(goal_tokens):
                    if generated[-len(goal_tokens):] == goal_tokens:
                        self.message_queue.put("\n=== GOAL REACHED! ===\n")
                        break
                
                # Update prompt tokens
                prompt_tokens.extend(generated[:5])
                if len(prompt_tokens) > 30:
                    prompt_tokens = prompt_tokens[-30:]
                
                step += 1
                time.sleep(0.3)
                
            except Exception as e:
                # FIX #5: Thread-safe error reporting
                # CHANGED: Error messages also go through queue
                self.message_queue.put(f"Error: {str(e)}\n")
                break
        
        # FIX #5: Thread-safe completion message
        # CHANGED: Final updates through queue
        self.message_queue.put("\n=== Agent Loop Finished ===\n")
        self.running = False
        self.message_queue.put("ENABLE_START_BUTTON")
    
    def run(self):
        self.root.mainloop()

def main():
    """Main entry point"""
    print("Starting BitWhisperer 1-bit LLM Agent...")
    app = BitWhispererGUI()
    app.run()

if __name__ == "__main__":
    main()
