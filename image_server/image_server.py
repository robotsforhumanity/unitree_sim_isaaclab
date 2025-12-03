# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A ZMQ-based image server that reads multi-image data from shared memory and publishes it
"""

import cv2
import zmq
import time
import threading
from image_server.shared_memory_utils import MultiImageReader, SharedMemoryReader


class ImageServer:
    def __init__(self, fps=30, port=55555, Unit_Test=False):
        """
        Multi-image server - read multi-image data from shared memory and publish it
        Port 55555 matches the default zmq_port in cam_config_server.yaml for head_camera
        """
        print("[Image Server] Initializing multi-image server from shared memory")
        
        self.fps = fps
        self.port = port
        self.Unit_Test = Unit_Test
        self.running = False
        self.publish_thread = None
        self.frame_count = 0

        # Initialize multi-image shared memory reader
        self.multi_image_reader = MultiImageReader()
        #self.image_depth_reader = SharedMemoryReader(shm_name="image_head_depth")
        #self.image_segmentation_reader = SharedMemoryReader(shm_name="image_head_segmentation")

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        print(f"[Image Server] Multi-image server initialized on port {self.port}")
        
        # start the publishing thread
        self.start_publishing()

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.time_window = 1.0
        self.frame_times = []
        self.start_time = time.time()

    def _update_performance_metrics(self, current_time):
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        self.frame_times = [t for t in self.frame_times if t >= current_time - self.time_window]
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window if self.frame_times else 0
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def send_process(self):
        """Read the concatenated images from shared memory and send them"""
        print("[Image Server] Starting send_process from shared memory...")
        
        try:
            while True:
                # read the concatenated images from shared memory
                concatenated_image = self.multi_image_reader.read_concatenated_image()
                
                if concatenated_image is None:
                    # if there is no image data, wait a moment and try again
                    time.sleep(0.01)
                    continue
                
                # show the concatenated images
                # cv2.imshow('Concatenated Images (Head + Left + Right)', concatenated_image)
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                #     print("[Image Server] User pressed quit key")
                #     break
                
                # encode the images
                ret, buffer = cv2.imencode('.jpg', concatenated_image)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                # build the message
                message = jpg_bytes

                # send the message
                self.socket.send(message)
               
                #image_depth = self.image_depth_reader.read_image()
                #if image_depth is not None:
                #    # normalizar para envío
                #    #depth_vis = cv2.normalize(image_depth, None, 0, 255, cv2.NORM_MINMAX)
                #    ret, buffer = cv2.imencode(".png", image_depth)
                #    jpg_bytes_depth = buffer.tobytes()
                #    #self.socket.send(jpg_bytes_depth)
                #    #cv2.imshow("Depth Image", d    epth_vis)
                #
                #image_segmentation = self.image_segmentation_reader.read_image()
                #if image_segmentation is not None:
                #    ret, buffer = cv2.imencode(".png", image_segmentation)
                #    jpg_bytes_segmentation = buffer.tobytes()
                #    #self.socket.send(jpg_bytes_segmentation)

                #self.socket.send_multipart([b'FRAME', jpg_bytes, jpg_bytes_depth,  jpg_bytes_segmentation])
                self.frame_count += 1

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            cv2.destroyAllWindows()  # close the display window
            self._close()

    def start_publishing(self):
        """Start the publishing thread"""
        if not self.running:
            self.running = True
            self.publish_thread = threading.Thread(target=self.send_process)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            print("[Image Server] Multi-image publishing thread started")

    def stop_publishing(self):
        """Stop the publishing thread"""
        if self.running:
            self.running = False
            if self.publish_thread:
                self.publish_thread.join(timeout=1.0)
            print("[Image Server] Publishing thread stopped")

    def _close(self):
        """Close the server"""
        self.stop_publishing()
        cv2.destroyAllWindows()
        
        # close the shared memory reader
        if hasattr(self, 'multi_image_reader'):
            self.multi_image_reader.close()
            #self.image_depth_reader.close()
            #self.image_segmentation_reader.close()
            
        # close the network connection
        self.socket.close()
        self.context.term()
        print("[Image Server] Multi-image server closed")

    def __del__(self):
        """Destructor"""
        self._close()


if __name__ == "__main__":
    # use the send_process mode example
    server = ImageServer(fps=30, Unit_Test=False)
    
    # use the send_process method (blocking)
    server.send_process()
    
    # or use the thread mode
    # try:
    #     server.start_publishing()
    #     print("[Image Server] Server running... Press Ctrl+C to stop")
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\n[Image Server] Interrupted by user")
    # finally:
    #     server._close()