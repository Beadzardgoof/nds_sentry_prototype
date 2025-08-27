import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import logging

class ThreadType(Enum):
    IMAGE_CAPTURE = "image_capture"
    YOLO_PROCESSOR = "yolo_processor"
    PREDICTION = "prediction"
    EXTRAPOLATION = "extrapolation"
    MASTER_CONTROLLER = "master_controller"
    STANDBY = "standby"

class ThreadState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    TERMINATING = "terminating"

class ThreadPoolManager:
    def __init__(self, max_threads: int = 20):
        self.max_threads = max_threads
        self.threads: Dict[str, Dict] = {}
        self.thread_counter = 0
        self.lock = threading.Lock()
        
        # Thread allocation limits
        self.thread_limits = {
            ThreadType.IMAGE_CAPTURE: 1,
            ThreadType.YOLO_PROCESSOR: 15,  # Dynamic, can be reduced
            ThreadType.PREDICTION: 4,
            ThreadType.EXTRAPOLATION: 2,
            ThreadType.MASTER_CONTROLLER: 1,
            ThreadType.STANDBY: 2
        }
        
        # Current allocations
        self.allocations = {thread_type: 0 for thread_type in ThreadType}
        
        # Failure rate tracking for prediction
        self.failure_history = deque(maxlen=100)  # 1 second at ~100 FPS
        self.last_failure_rate = 0.0
        
        self.logger = logging.getLogger(__name__)

    def get_thread_id(self) -> str:
        with self.lock:
            self.thread_counter += 1
            return f"thread_{self.thread_counter:04d}"

    def create_thread(self, thread_type: ThreadType, target: Callable, 
                     args: tuple = (), kwargs: dict = None) -> Optional[str]:
        if kwargs is None:
            kwargs = {}
            
        with self.lock:
            if len(self.threads) >= self.max_threads:
                self.logger.warning(f"Thread pool full, cannot create {thread_type}")
                return None
                
            if self.allocations[thread_type] >= self.thread_limits[thread_type]:
                self.logger.warning(f"Thread limit reached for {thread_type}")
                return None
                
            thread_id = self.get_thread_id()
            
            thread = threading.Thread(
                target=target,
                args=args,
                kwargs=kwargs,
                name=f"{thread_type.value}_{thread_id}",
                daemon=True
            )
            
            self.threads[thread_id] = {
                'thread': thread,
                'type': thread_type,
                'state': ThreadState.IDLE,
                'created_at': time.time(),
                'target': target,
                'args': args,
                'kwargs': kwargs
            }
            
            self.allocations[thread_type] += 1
            thread.start()
            
            self.logger.info(f"Created thread {thread_id} of type {thread_type}")
            return thread_id

    def terminate_thread(self, thread_id: str) -> bool:
        with self.lock:
            if thread_id not in self.threads:
                return False
                
            thread_info = self.threads[thread_id]
            thread_info['state'] = ThreadState.TERMINATING
            
            # Signal thread to stop (implementation depends on target function)
            thread_info['thread'].join(timeout=1.0)
            
            # Update allocations
            self.allocations[thread_info['type']] -= 1
            
            # Remove from pool
            del self.threads[thread_id]
            
            self.logger.info(f"Terminated thread {thread_id}")
            return True

    def get_threads_by_type(self, thread_type: ThreadType) -> List[str]:
        with self.lock:
            return [tid for tid, info in self.threads.items() 
                   if info['type'] == thread_type]

    def get_available_standby_threads(self) -> List[str]:
        return self.get_threads_by_type(ThreadType.STANDBY)

    def convert_thread_type(self, thread_id: str, new_type: ThreadType, 
                           new_target: Callable, args: tuple = (), kwargs: dict = None) -> bool:
        if kwargs is None:
            kwargs = {}
            
        with self.lock:
            if thread_id not in self.threads:
                return False
                
            if self.allocations[new_type] >= self.thread_limits[new_type]:
                self.logger.warning(f"Cannot convert to {new_type}, limit reached")
                return False
                
            old_info = self.threads[thread_id]
            old_type = old_info['type']
            
            # Terminate old thread
            old_info['state'] = ThreadState.TERMINATING
            old_info['thread'].join(timeout=1.0)
            
            # Create new thread with same ID
            new_thread = threading.Thread(
                target=new_target,
                args=args,
                kwargs=kwargs,
                name=f"{new_type.value}_{thread_id}",
                daemon=True
            )
            
            # Update allocations
            self.allocations[old_type] -= 1
            self.allocations[new_type] += 1
            
            # Update thread info
            self.threads[thread_id] = {
                'thread': new_thread,
                'type': new_type,
                'state': ThreadState.IDLE,
                'created_at': time.time(),
                'target': new_target,
                'args': args,
                'kwargs': kwargs
            }
            
            new_thread.start()
            
            self.logger.info(f"Converted thread {thread_id} from {old_type} to {new_type}")
            return True

    def update_failure_rate(self, success: bool):
        """Update failure rate tracking for predictive mode switching"""
        self.failure_history.append(not success)  # Store failures, not successes
        
        if len(self.failure_history) > 0:
            self.last_failure_rate = sum(self.failure_history) / len(self.failure_history)

    def predict_failure_rate_threshold(self) -> bool:
        """Predict if failure rate will exceed 0.8 in next 0.2 seconds"""
        if len(self.failure_history) < 20:  # Need at least 0.2s of data
            return False
            
        # Calculate historical rate over last 1 second (or available data)
        recent_failures = list(self.failure_history)[-100:]  # Last 1 second max
        historical_rate = sum(recent_failures) / len(recent_failures)
        
        # Prediction: (historical_rate * 0.2) + current_rate > 0.8
        predicted_rate = (historical_rate * 0.2) + self.last_failure_rate
        
        return predicted_rate > 0.8

    def should_switch_to_extrapolation(self) -> bool:
        """Check if we should switch to extrapolation mode"""
        return self.predict_failure_rate_threshold()

    def should_switch_to_prediction(self) -> bool:
        """Check if we should switch back to prediction mode (low failure for 1s)"""
        if len(self.failure_history) < 100:  # Need full 1 second of data
            return False
            
        # Check if failure rate has been low for at least 1 second
        recent_failures = list(self.failure_history)[-100:]
        recent_failure_rate = sum(recent_failures) / len(recent_failures)
        
        return recent_failure_rate < 0.2  # Low failure rate

    def rebalance_threads(self):
        """Dynamically rebalance thread allocation based on current needs"""
        with self.lock:
            # Check if mode switch is needed
            if self.should_switch_to_extrapolation():
                self._switch_to_extrapolation_mode()
            elif self.should_switch_to_prediction():
                self._switch_to_prediction_mode()
                
            # Maximize YOLO threads with remaining capacity
            self._maximize_yolo_threads()

    def _switch_to_extrapolation_mode(self):
        """Switch from prediction to extrapolation threads"""
        prediction_threads = self.get_threads_by_type(ThreadType.PREDICTION)
        standby_threads = self.get_available_standby_threads()
        
        if prediction_threads and standby_threads:
            # Use standby thread for extrapolation, then terminate prediction
            standby_id = standby_threads[0]
            # Convert standby to extrapolation (implementation specific)
            # Then terminate one prediction thread
            if prediction_threads:
                self.terminate_thread(prediction_threads[0])
                
        self.logger.info("Switched to extrapolation mode")

    def _switch_to_prediction_mode(self):
        """Switch from extrapolation to prediction threads"""
        extrapolation_threads = self.get_threads_by_type(ThreadType.EXTRAPOLATION)
        standby_threads = self.get_available_standby_threads()
        
        if extrapolation_threads and standby_threads:
            # Use standby thread for prediction, then terminate extrapolation
            standby_id = standby_threads[0]
            # Convert standby to prediction (implementation specific)
            # Then terminate extrapolation thread
            if extrapolation_threads:
                self.terminate_thread(extrapolation_threads[0])
                
        self.logger.info("Switched to prediction mode")

    def _maximize_yolo_threads(self):
        """Allocate remaining threads to YOLO processing"""
        current_total = sum(self.allocations.values())
        available_slots = self.max_threads - current_total
        
        # Convert available standby threads to YOLO if we have capacity
        standby_threads = self.get_available_standby_threads()
        yolo_threads = self.get_threads_by_type(ThreadType.YOLO_PROCESSOR)
        
        max_additional_yolo = min(
            available_slots, 
            self.thread_limits[ThreadType.YOLO_PROCESSOR] - len(yolo_threads),
            len(standby_threads) - 2  # Keep at least 2 standby
        )
        
        for i in range(max_additional_yolo):
            if i < len(standby_threads) - 2:
                standby_id = standby_threads[i]
                # Convert to YOLO (implementation specific)

    def get_status(self) -> Dict[str, Any]:
        """Get current thread pool status"""
        with self.lock:
            status = {
                'total_threads': len(self.threads),
                'max_threads': self.max_threads,
                'allocations': dict(self.allocations),
                'failure_rate': self.last_failure_rate,
                'prediction_threshold': self.predict_failure_rate_threshold(),
                'active_threads': {
                    thread_type.value: [tid for tid, info in self.threads.items() 
                                      if info['type'] == thread_type]
                    for thread_type in ThreadType
                }
            }
        return status

    def shutdown(self):
        """Shutdown all threads"""
        with self.lock:
            thread_ids = list(self.threads.keys())
            for thread_id in thread_ids:
                self.terminate_thread(thread_id)
            
        self.logger.info("Thread pool shutdown complete")