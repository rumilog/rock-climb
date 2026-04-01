#!/usr/bin/env python3
"""
LEAP Hand Teleoperation from Quest 2

This script receives finger angle data forwarded from server_env.py (localhost:8002)
and controls the LEAP hand to mirror Quest 2 hand tracking movements.

Architecture:
- Quest 2 sends UDP data to server_env.py (port 8001)
- server_env.py forwards finger data to this script (localhost:8002)
- This script controls the LEAP hand motors

Note: server_env.py must be running for this script to receive data.
"""

import numpy as np
import time
import sys
import signal
import threading
import tty
import termios
import select
from datetime import datetime

# Add the LEAP hand API to the path
import sys
sys.path.append('LEAP_Hand_API/python')

from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu
import UdpComms as U

# Configuration
THIS_IP = "172.26.67.113"  # Your PC's IP
OCULUS_IP = "172.26.84.138"  # Quest 2's IP
LEAP_PORT = None  # Auto-detect; set to e.g. '/dev/ttyUSB0' to override


def find_leap_port():
    """Scan /dev/ttyUSB* ports and return the first one that opens at 4 Mbaud,
    or None if no suitable port is found."""
    import glob
    candidates = sorted(glob.glob('/dev/ttyUSB*'))
    for port in candidates:
        try:
            import serial
            s = serial.Serial(port, 4000000, timeout=0.05)
            s.close()
            return port
        except Exception:
            continue
    # Fallback: if pyserial not available, just return the first candidate
    return candidates[0] if candidates else None


class LeapPipDipTeleop:
    def __init__(self, port=LEAP_PORT, verbose=True):
        """Initialize LEAP Hand controller for PIP/DIP teleoperation"""
        self.verbose = verbose
        if port is None:
            port = find_leap_port()
            if port is None:
                print("⚠️  No ttyUSB device found — LEAP hand will be offline")
                port = '/dev/ttyUSB0'  # fallback for error messages
        self.port = port
        self.dxl_client = None
        self.motors = list(range(16))
        self.running = False
        self._consecutive_errors = 0
        self._max_errors_before_reconnect = 10
        self._reconnecting = False
        
        # LEAP hand configuration
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        
        # Initialize LEAP hand
        self.init_leap_hand()
        
        # Initialize UDP communication
        self.init_udp()
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # Scaling to increase closing for PIP/DIP (method 2)
        # Tune these if fist is under-closing
        self.pip_scale = 2
        self.dip_scale = 2
        self.mcp_flex_scale = 1.0
        self.mcp_abd_scale = 1.0
        # Per-finger multipliers [Index, Middle, Pinky, Thumb]
        # Note: these multiply with the global scales above (e.g. pip_scale * pip_scale_per_finger)
        self.pip_scale_per_finger = [1.0, 1.15, 1.4, 1.7]
        self.dip_scale_per_finger = [1.0, 1.10, 1.5, 1.8]
        self.mcp_flex_scale_per_finger = [1.2, 1.6, 2, 5]
        self.mcp_abd_scale_per_finger = [1.4, 1.2, 1.4, 5]
        # Per-finger zero-offsets (degrees) to treat measured straight/neutral as 0
        # Set to the EXACT measured value when the joint is straight (can be negative!)
        # For example, if "straight" measures -20°, set offset to -20.0
        # Order: [Index, Middle, Pinky, Thumb]
        self.dip_zero_offset_deg_per_finger = [0.0, 0.0, 15.0, 50.0]
        self.pip_zero_offset_deg_per_finger = [0.0, 0.0, 25.0, 40.0]
        self.mcp_flex_zero_offset_deg_per_finger = [0.0, 0.0, 0.0, 40.0]
        self.mcp_abd_zero_offset_deg_per_finger = [10.0, 0.0, -10.0, 30.0]
        
        # Post-scaling offsets (in RADIANS) - applied AFTER conversion and scaling
        # Simple addition/subtraction to final motor commands for fine-tuning
        # Order: [Index, Middle, Pinky, Thumb]
        self.dip_post_scale_offset_rad_per_finger = [0.0, 0.0, 0.0, 0.0]
        self.pip_post_scale_offset_rad_per_finger = [0.0, 0.0, 0.0, 0.3]
        self.mcp_flex_post_scale_offset_rad_per_finger = [0.0, 0.0, 0.0, -0.7]
        self.mcp_abd_post_scale_offset_rad_per_finger = [0, 0.1, 0.2, 0.0]

        # Thumb joint tuning (arrow keys)
        # Joints: 0=MCP_Abd(12), 1=MCP_Flex(13), 2=PIP(14), 3=DIP(15)
        self.thumb_joint_names = ["MCP_Abd", "MCP_Flex", "PIP", "DIP"]
        self.thumb_selected = 1          # start on MCP_Flex
        # MCP_Flex (index 1) starts at +0.9 to clear the -1.3 post-scale offset
        # that otherwise pins it at the hardware clip minimum (-0.47 allegro).
        self.thumb_offsets = np.array([0.0, 0.9, 0.0, 0.0])
        self.thumb_step = 0.05           # ~3° per keypress

    def init_leap_hand(self):
        """Initialize LEAP hand connection"""
        try:
            if self.verbose:
                print("🔌 Connecting to LEAP hand...")

            # Create client
            self.dxl_client = DynamixelClient(self.motors, self.port, 4000000)
            self.dxl_client.connect()

            if self.verbose:
                print("⚙️ Configuring LEAP hand...")
            # Must disable torque before changing operating mode (Dynamixel requirement)
            self.dxl_client.set_torque_enabled(self.motors, False, retries=5)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors))*5, 11, 1)
            self.dxl_client.set_torque_enabled(self.motors, True, retries=5)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
            self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
            self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

            # Initialize to open position
            open_positions = np.zeros(16)
            leap_positions = lhu.allegro_to_LEAPhand(open_positions)
            self.dxl_client.write_desired_pos(self.motors, leap_positions)
            self.last_read_pos = self.dxl_client.read_pos()

            if self.verbose:
                print("✅ LEAP Hand connected and configured")
                print("📋 Joint order: Index[0-3], Middle[4-7], Pinky[8-11], Thumb[12-15]")
                print("📋 Each finger: [MCP Side, MCP Forward, PIP, DIP]")

        except Exception as e:
            print(f"❌ Failed to connect to LEAP hand: {e}")
            print(f"⚠️  Continuing in simulation mode...")
            self.dxl_client = None
    
    def init_udp(self):
        """Initialize UDP communication - receives forwarded finger data from VR_Teleoperation_Minimum.py"""
        try:
            if self.verbose:
                print("📡 Setting up UDP communication...")
                print(f"   Listening on: 127.0.0.1:8002 (forwarded from VR_Teleoperation_Minimum.py)")

            self.sock = U.UdpComms(udpIP="127.0.0.1", sendIP="127.0.0.1", portTX=8002, portRX=8002,
                                   enableRX=True, suppressWarnings=True)

            if self.verbose:
                print("✅ UDP communication ready")

        except Exception as e:
            print(f"❌ UDP setup failed: {e}")
            self.sock = None

    def reconnect(self):
        """Attempt to reconnect to the LEAP hand on any available ttyUSB port."""
        if self._reconnecting:
            return False
        self._reconnecting = True
        print("\n🔄 LEAP hand connection lost — scanning for new port...")

        # Close the old connection gracefully
        if self.dxl_client is not None:
            try:
                self.dxl_client.port_handler.closePort()
            except Exception:
                pass
            self.dxl_client = None

        # Scan for the new port (USB re-enumeration may take a moment)
        new_port = None
        for attempt in range(10):
            new_port = find_leap_port()
            if new_port is not None:
                break
            print(f"   Waiting for USB device... ({attempt + 1}/10)")
            time.sleep(1.0)

        if new_port is None:
            print("❌ No ttyUSB device found after 10s. LEAP hand offline.")
            self._reconnecting = False
            return False

        print(f"   Found port: {new_port}")
        self.port = new_port

        try:
            self.dxl_client = DynamixelClient(self.motors, self.port, 4000000)
            self.dxl_client.connect()
            self.dxl_client.set_torque_enabled(self.motors, False)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
            self.dxl_client.set_torque_enabled(self.motors, True)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
            self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
            self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
            self._consecutive_errors = 0
            print(f"✅ Reconnected to LEAP hand on {new_port}")
            self._reconnecting = False
            return True
        except Exception as e:
            print(f"❌ Reconnection failed: {e}")
            self.dxl_client = None
            self._reconnecting = False
            return False

    def update_pip_dip_joints(self, quest_angles_rad):
        """
        Update only PIP and DIP joints based on Quest 2 tracking data
        
        Args:
            quest_angles_rad: 8 values in radians [Thumb PIP, Thumb DIP, 
                                                   Index PIP, Index DIP,
                                                   Middle PIP, Middle DIP, 
                                                   Pinky PIP, Pinky DIP]
        """
        if not self.dxl_client:
            return False

    def update_full_joints_deg16(self, angles_deg16):
        """
        Update MCP_Abd, MCP_Flex, PIP, DIP for Index, Middle, Pinky, Thumb.
        Input: 16 values in DEGREES ordered per frame: 
        [Index_DIP, Index_PIP, Index_MCP_Flex, Index_MCP_Abd,
         Middle_DIP, Middle_PIP, Middle_MCP_Flex, Middle_MCP_Abd,
         Pinky_DIP, Pinky_PIP, Pinky_MCP_Flex, Pinky_MCP_Abd,
         Thumb_DIP, Thumb_PIP, Thumb_MCP_Flex, Thumb_MCP_Abd]
        """
        if not self.dxl_client:
            if self._consecutive_errors >= self._max_errors_before_reconnect:
                self.reconnect()
                self._consecutive_errors = 0
            return False
        try:
            # Pinky data (indices 8-11) now comes directly from the Quest's
            # pinky finger tracking — no override needed.
            angles_deg16[10] = angles_deg16[6]  # MCP_Flex
            # Apply zero-offsets in degrees before conversion/scaling
            angles_adj = np.array(angles_deg16, dtype=float)
            for f in range(4):
                base = f * 4
                angles_adj[base + 0] = angles_adj[base + 0] - self.dip_zero_offset_deg_per_finger[f]      # DIP
                angles_adj[base + 1] = angles_adj[base + 1] - self.pip_zero_offset_deg_per_finger[f]      # PIP
                angles_adj[base + 2] = angles_adj[base + 2] - self.mcp_flex_zero_offset_deg_per_finger[f] # MCP_Flex
                angles_adj[base + 3] = angles_adj[base + 3] - self.mcp_abd_zero_offset_deg_per_finger[f] # MCP_Abd

            vals_rad = np.radians(angles_adj)
            # Debug: Print thumb MCP flex values (degrees) before and after offset
            # Thumb MCP flex is at index 12 + 2 = 14 in angles_deg16, or base=12, offset=2 in angles_adj
            # Temporarily enable this to debug offset issues - check what values you're getting
            if False:  # Set to True to enable debug
                thumb_mcp_flex_orig = angles_deg16[12 + 2] if len(angles_deg16) > 14 else 0
                thumb_mcp_flex_after_offset = angles_adj[12 + 2] if len(angles_adj) > 14 else 0
                thumb_mcp_flex_rad_before_scale = vals_rad[12 + 2] if len(vals_rad) > 14 else 0
                print(f"Thumb MCP Flex: orig={thumb_mcp_flex_orig:.2f}°, after_offset={thumb_mcp_flex_after_offset:.2f}°, offset={self.mcp_flex_zero_offset_deg_per_finger[3]}, rad_before_scale={thumb_mcp_flex_rad_before_scale:.4f}")
            # Apply scaling to all joints for each finger block
            # Order: [DIP +0, PIP +1, MCP_Flex +2, MCP_Abd +3]
            for f in range(4):
                base = f * 4
                vals_rad[base + 0] *= (self.dip_scale * self.dip_scale_per_finger[f])  # DIP
                vals_rad[base + 1] *= (self.pip_scale * self.pip_scale_per_finger[f])  # PIP
                vals_rad[base + 2] *= (self.mcp_flex_scale * self.mcp_flex_scale_per_finger[f])  # MCP_Flex
                vals_rad[base + 3] *= (self.mcp_abd_scale * self.mcp_abd_scale_per_finger[f])  # MCP_Abd
            
            # Apply post-scaling offsets (in radians) - simple addition/subtraction
            for f in range(4):
                base = f * 4
                vals_rad[base + 0] += self.dip_post_scale_offset_rad_per_finger[f]      # DIP
                vals_rad[base + 1] += self.pip_post_scale_offset_rad_per_finger[f]      # PIP
                vals_rad[base + 2] += self.mcp_flex_post_scale_offset_rad_per_finger[f] # MCP_Flex
                vals_rad[base + 3] += self.mcp_abd_post_scale_offset_rad_per_finger[f] # MCP_Abd
            
            # Read current LEAP (allegro convention)
            current_leap = self.dxl_client.read_pos()
            self.last_read_pos = current_leap.copy()
            current_allegro = lhu.LEAPhand_to_allegro(current_leap, zeros=False)

            def map_finger(block_start, dip_idx):
                # block is [DIP, PIP, MCP_Flex, MCP_Abd]
                DIP = vals_rad[dip_idx + 0]
                PIP = vals_rad[dip_idx + 1]
                MCP_F = vals_rad[dip_idx + 2]
                MCP_A = vals_rad[dip_idx + 3]
                # Invert MCP abduction to match LEAP hand kinematics
                # Index starts at 0, Middle at 4, Pinky at 8; do NOT invert Thumb (12)
                if block_start in (0, 4, 8):
                    MCP_A = -MCP_A
                # Invert Thumb DIP (backwards)
                if block_start == 12:
                    DIP = -DIP
                current_allegro[block_start + 0] = MCP_A
                current_allegro[block_start + 1] = MCP_F
                current_allegro[block_start + 2] = PIP
                current_allegro[block_start + 3] = DIP

            # Index block start 0, packet offset 0
            map_finger(0, 0)
            # Middle block start 4, packet offset 4
            map_finger(4, 4)
            # Pinky block start 8, packet offset 8
            map_finger(8, 8)
            # Thumb block start 12, packet offset 12
            map_finger(12, 12)

            # Apply per-joint thumb offsets (arrow key tuning)
            for i in range(4):
                current_allegro[12 + i] += self.thumb_offsets[i]

            leap_positions = lhu.allegro_to_LEAPhand(current_allegro, zeros=False)
            leap_positions = lhu.angle_safety_clip(leap_positions)

            if not hasattr(self, '_debug_write_count'):
                self._debug_write_count = 0
                self._prev_current = None
            self._debug_write_count += 1

            if self.verbose:
                show_dbg = (self._debug_write_count <= 5 or
                            self._debug_write_count % 200 == 0)
                if show_dbg:
                    changed = "CHANGED" if (self._prev_current is None or
                        not np.allclose(current_leap, self._prev_current, atol=1e-4)) else "FROZEN"
                    print(f"  [LEAP DBG] write #{self._debug_write_count}: "
                          f"max_delta={np.abs(leap_positions - current_leap).max():.4f} rad, "
                          f"current_status={changed}, "
                          f"target[0:4]={np.round(leap_positions[:4], 3)}, "
                          f"current[0:4]={np.round(current_leap[:4], 3)}")
            self._prev_current = current_leap.copy()

            self.dxl_client.write_desired_pos(self.motors, leap_positions)

            if self.verbose and self._debug_write_count <= 5:
                time.sleep(0.02)
                verify = self.dxl_client.read_pos()
                moved = np.abs(verify - current_leap).max()
                print(f"  [VERIFY] After write #{self._debug_write_count}: "
                      f"motor moved {moved:.4f} rad in 20ms")

            self._consecutive_errors = 0
            return True
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors == 1 or self._consecutive_errors % 50 == 0:
                print(f"⚠️  Dynamixel error ({self._consecutive_errors}x): {e}")
            if self._consecutive_errors >= self._max_errors_before_reconnect:
                self.reconnect()
            return False
        
        try:
            # Get current LEAP hand positions (in Allegro convention)
            current_leap = self.dxl_client.read_pos()
            current_allegro = lhu.LEAPhand_to_allegro(current_leap, zeros=False)
            
            # LEAP hand joint indices:
            # Index: 0=MCP_Side, 1=MCP_Flex, 2=PIP, 3=DIP
            # Middle: 4=MCP_Side, 5=MCP_Flex, 6=PIP, 7=DIP
            # Pinky: 8=MCP_Side, 9=MCP_Flex, 10=PIP, 11=DIP
            # Thumb: 12=MCP_Side, 13=MCP_Flex, 14=PIP, 15=DIP
            
            # Quest order: [Thumb PIP, Thumb DIP, Index PIP, Index DIP,
            #              Middle PIP, Middle DIP, Pinky PIP, Pinky DIP]
            #              [0         , 1         , 2        , 3       ,
            #               4         , 5         , 6        , 7]
            
            # Update only PIP and DIP joints, keep MCP joints unchanged
            
            # Thumb (Quest indices 0,1) -> LEAP joints 14,15
            current_allegro[14] = quest_angles_rad[0]  # Thumb PIP
            current_allegro[15] = quest_angles_rad[1]  # Thumb DIP
            
            # Index (Quest indices 2,3) -> LEAP joints 2,3
            current_allegro[2] = quest_angles_rad[2]  # Index PIP
            current_allegro[3] = quest_angles_rad[3]  # Index DIP
            
            # Middle (Quest indices 4,5) -> LEAP joints 6,7
            current_allegro[6] = quest_angles_rad[4]  # Middle PIP
            current_allegro[7] = quest_angles_rad[5]  # Middle DIP
            
            # Pinky (Quest indices 6,7) -> LEAP joints 10,11
            current_allegro[10] = quest_angles_rad[6]  # Pinky PIP
            current_allegro[11] = quest_angles_rad[7]  # Pinky DIP
            
            # Note: LEAP hand has 4 fingers: Index, Middle, Pinky, Thumb
            
            # Convert to LEAP coordinates and apply safety limits
            leap_positions = lhu.allegro_to_LEAPhand(current_allegro, zeros=False)
            leap_positions = lhu.angle_safety_clip(leap_positions)
            
            # Send to LEAP hand
            self.dxl_client.write_desired_pos(self.motors, leap_positions)
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating joints: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_udp_data(self, data):
        """Process incoming finger data forwarded from server_env.py"""
        try:
            if data is None:
                return False
            
            # Data is just the gripper_message string (finger angles) forwarded from server_env.py
            # Remove trailing tab if present
            gripper_message = data[:-1] if data.endswith('\t') else data
            
            if not gripper_message or gripper_message == "":
                return False
                    
            # Parse finger angles
            gripper_values = np.array(gripper_message.split('\t')).astype(np.float64)

            if not hasattr(self, '_udp_dbg_count'):
                self._udp_dbg_count = 0
            self._udp_dbg_count += 1
            if self.verbose and self._udp_dbg_count <= 3:
                print(f"  [UDP DBG] msg #{self._udp_dbg_count}: "
                      f"{len(gripper_values)} values, "
                      f"first 4: {gripper_values[:4]}, "
                      f"raw: {gripper_message[:80]}...")

            # If 28 values (16 angles + 12 thumb quaternions), ignore the last 12
            if len(gripper_values) == 28:
                gripper_values = gripper_values[:16]
            
            # Check if we have 10 values (PIP and DIP for 5 fingers)
            if len(gripper_values) == 10:
                # Update LEAP hand with PIP and DIP angles
                success = self.update_pip_dip_joints(gripper_values)
                return success
            elif len(gripper_values) == 16:
                # New format: 16 values in degrees (Index, Middle, Pinky, Thumb × DIP,PIP,MCP_Flex,MCP_Abd)
                success = self.update_full_joints_deg16(gripper_values)
                return success
            else:
                if len(gripper_values) != 0:  # Don't print for empty messages
                    print(f"⚠️  Expected 10 (PIP/DIP) or 16 (full) values, got {len(gripper_values)}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing UDP data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _keyboard_thread(self):
        """Read up/down arrow keys to adjust thumb abduction offset."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        seq = sys.stdin.read(2)
                        if seq == '[A':   # Up arrow
                            self.thumb_abd_offset += self.thumb_abd_step
                        elif seq == '[B': # Down arrow
                            self.thumb_abd_offset -= self.thumb_abd_step
                        else:
                            continue
                        print(f"\r  [THUMB ABD] offset: {self.thumb_abd_offset:+.3f} rad "
                              f"({np.degrees(self.thumb_abd_offset):+.1f}°)    ", flush=True)
                    elif ch in ('\x03', 'q'):  # Ctrl+C or q
                        self.running = False
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run_teleoperation(self):
        """Main teleoperation loop"""
        print("\n" + "="*60)
        print("🚀 LEAP HAND TELEOPERATION")
        print("="*60)
        print("💡 Move your hand to control the LEAP hand")
        print("📡 Receiving data from server_env.py (localhost:8002)")
        print("⬆⬇  Up/Down arrows: adjust thumb abduction offset")
        print("⚠️  Make sure server_env.py is running!")
        print("⚠️  Press Ctrl+C or q to stop")
        print("="*60 + "\n")

        self.running = True
        kb_thread = threading.Thread(target=self._keyboard_thread, daemon=True)
        kb_thread.start()
        frame_count = 0
        last_print_time = time.time()
        
        try:
            while self.running:
                # Receive data from Quest 2
                data = self.sock.ReadReceivedData()
                
                if data is not None:
                    success = self.process_udp_data(data)
                    
                    if success:
                        frame_count += 1
                        
                        # Print status every second
                        current_time = time.time()
                        if current_time - last_print_time >= 1.0:
                            fps = frame_count / (current_time - last_print_time)
                            print(f"✓ Teleoperating @ {fps:.1f} Hz (total frames: {frame_count})")
                            frame_count = 0
                            last_print_time = current_time
                
                # Small sleep to avoid busy waiting
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Teleoperation stopped by user")
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🔄 Cleaning up...")
        
        if self.dxl_client:
            try:
                # Return to open position
                print("🖐️  Resetting LEAP hand to open position...")
                open_positions = np.zeros(16)
                leap_positions = lhu.allegro_to_LEAPhand(open_positions)
                self.dxl_client.write_desired_pos(self.motors, leap_positions)
                time.sleep(0.5)
                
                # Disconnect
                self.dxl_client.set_torque_enabled(self.motors, False)
                self.dxl_client.disconnect()
                
                print("✅ LEAP hand safely disconnected")
                
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")
        
        self.running = False
        print("✓ Teleoperation stopped")
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Interrupt received! Stopping teleoperation...")
        self.running = False

def main():
    try:
        teleop = LeapPipDipTeleop()
        
        if not teleop.sock:
            print("❌ UDP communication not available")
            return
        
        if not teleop.dxl_client:
            print("⚠️  LEAP hand not connected - running in simulation mode")
        
        # Start teleoperation
        teleop.run_teleoperation()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

