# Copyright (c) 2025, Robots For Humanity
# Simple shared memory communication for recording commands
# This bypasses DDS to ensure reliable inter-process communication

import mmap
import os
import time
from typing import Optional

class RecordingCommandShm:
    """Shared memory for recording commands (START_REC, STOP_REC)"""
    
    SHM_NAME = "/tmp/recording_cmd_shm"
    SHM_SIZE = 64  # bytes
    
    def __init__(self, is_writer: bool = False):
        """
        Initialize shared memory for recording commands.
        
        Args:
            is_writer: True for teleoperator (writes commands), False for simulator (reads commands)
        """
        self.is_writer = is_writer
        self.shm_file = None
        self.mmap_obj = None
        
        # Create or open the shared memory file
        try:
            # Both reader and writer create the file if it doesn't exist
            if not os.path.exists(self.SHM_NAME):
                # Create the file if it doesn't exist
                with open(self.SHM_NAME, "wb") as f:
                    f.write(b'\x00' * self.SHM_SIZE)
                print(f"[RecordingCommandShm] Created shared memory file: {self.SHM_NAME}")
            
            # Now open it for reading and writing
            self.shm_file = open(self.SHM_NAME, "r+b")
            
            if self.shm_file:
                access = mmap.ACCESS_WRITE if self.is_writer else mmap.ACCESS_READ
                self.mmap_obj = mmap.mmap(self.shm_file.fileno(), self.SHM_SIZE, access=access)
                print(f"[RecordingCommandShm] Initialized as {'writer' if is_writer else 'reader'}")
                if self.is_writer:
                    self._clear_buffer()
                    print("[RecordingCommandShm] Writer cleared shared memory buffer.")
        except Exception as e:
            print(f"[RecordingCommandShm] Failed to initialize: {e}")
    
    def write_command(self, command: str):
        """Write a recording command to shared memory (teleoperator side)"""
        if not self.is_writer or not self.mmap_obj:
            return
        
        try:
            # Encode command as UTF-8, padded/truncated to SHM_SIZE
            cmd_bytes = command.encode('utf-8')[:self.SHM_SIZE-1]
            cmd_bytes = cmd_bytes + b'\x00' * (self.SHM_SIZE - len(cmd_bytes))
            
            self.mmap_obj.seek(0)
            self.mmap_obj.write(cmd_bytes)
            self.mmap_obj.flush()
            if self.shm_file:
                self.shm_file.flush()
                os.fsync(self.shm_file.fileno())
        except Exception as e:
            print(f"[RecordingCommandShm] Failed to write command: {e}")

    def _clear_buffer(self):
        """Internal helper to zero the shared memory region."""
        if not self.mmap_obj:
            return
        self.mmap_obj.seek(0)
        self.mmap_obj.write(b'\x00' * self.SHM_SIZE)
        self.mmap_obj.flush()
        if self.shm_file:
            self.shm_file.flush()
            os.fsync(self.shm_file.fileno())
    
    def read_command(self) -> Optional[str]:
        """Read the current recording command from shared memory (simulator side)"""
        if self.is_writer or not self.mmap_obj:
            return None
        
        try:
            self.mmap_obj.seek(0)
            cmd_bytes = self.mmap_obj.read(self.SHM_SIZE)
            # Decode and strip null bytes
            command = cmd_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
            return command if command else None
        except Exception as e:
            print(f"[RecordingCommandShm] Failed to read command: {e}")
            return None
    
    def close(self):
        """Close shared memory"""
        try:
            if self.mmap_obj:
                self.mmap_obj.close()
            if self.shm_file:
                self.shm_file.close()
            # NOTE: Do NOT delete the file - let it persist for other processes to use
            # The writer creates it once, and both reader/writer can use it
        except Exception as e:
            print(f"[RecordingCommandShm] Failed to close: {e}")
    
    def __del__(self):
        self.close()

