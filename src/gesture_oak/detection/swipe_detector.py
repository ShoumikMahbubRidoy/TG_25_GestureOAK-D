#!/usr/bin/env python3

import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple, List


class SwipeState(Enum):
    """スワイプ検出の状態"""
    IDLE = "idle"
    DETECTING = "detecting"
    VALIDATING = "validating"
    CONFIRMED = "confirmed"


class SwipeDetector:
    """左から右へのスワイプ検出クラス"""
    
    def __init__(self, 
                 buffer_size: int = 15,
                 min_distance: int = 120,
                 min_duration: float = 0.3,
                 max_duration: float = 2.0,
                 min_velocity: float = 50,
                 max_velocity: float = 800,
                 max_y_deviation: float = 0.3):
        """
        Args:
            buffer_size: 位置履歴のバッファサイズ
            min_distance: 最小移動距離 (px)
            min_duration: 最小継続時間 (秒)
            max_duration: 最大継続時間 (秒)
            min_velocity: 最小速度 (px/秒)
            max_velocity: 最大速度 (px/秒)
            max_y_deviation: Y方向の最大偏差率 (0.0-1.0)
        """
        self.buffer_size = buffer_size
        self.min_distance = min_distance
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.max_y_deviation = max_y_deviation
        
        # 状態管理
        self.state = SwipeState.IDLE
        self.position_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        
        # スワイプ検出用変数
        self.swipe_start_time = None
        self.swipe_start_pos = None
        self.last_confirmed_swipe = 0
        self.swipe_cooldown = 1.0  # スワイプ間のクールダウン時間(秒)
        
        # デバッグ用統計
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0
    
    def update(self, hand_center: Optional[Tuple[float, float]]) -> bool:
        """
        手の中心位置を更新してスワイプを検出
        
        Args:
            hand_center: 手の中心座標 (x, y) または None
            
        Returns:
            bool: スワイプが検出された場合True
        """
        current_time = time.time()
        
        # 手が検出されていない場合
        if hand_center is None:
            self._reset_detection()
            return False
        
        # 位置と時刻をバッファに追加
        self.position_buffer.append(hand_center)
        self.time_buffer.append(current_time)
        
        # 十分なデータが溜まっていない場合
        if len(self.position_buffer) < 3:
            return False
        
        # 状態に応じた処理
        if self.state == SwipeState.IDLE:
            return self._check_start_detection()
        elif self.state == SwipeState.DETECTING:
            return self._process_detection()
        elif self.state == SwipeState.VALIDATING:
            return self._validate_swipe()
        
        return False
    
    def _check_start_detection(self) -> bool:
        """スワイプ検出の開始条件をチェック"""
        if len(self.position_buffer) < 3:
            return False
        
        # 最近の3点で右方向の動きが検出されたら検出開始
        recent_positions = list(self.position_buffer)[-3:]
        
        # 連続する右方向の移動をチェック
        dx1 = recent_positions[1][0] - recent_positions[0][0]
        dx2 = recent_positions[2][0] - recent_positions[1][0]
        
        # 両方とも右方向で、かつ一定以上の移動量（より緩和）
        if dx1 > 3 and dx2 > 3:  # 検出開始の閾値を下げる
            self.state = SwipeState.DETECTING
            self.swipe_start_time = self.time_buffer[-3]
            self.swipe_start_pos = recent_positions[0]
        
        return False
    
    def _process_detection(self) -> bool:
        """スワイプ検出中の処理"""
        current_pos = self.position_buffer[-1]
        current_time = self.time_buffer[-1]
        
        # タイムアウトチェック
        if current_time - self.swipe_start_time > self.max_duration:
            self._reset_detection()
            return False
        
        # 移動距離チェック
        total_dx = current_pos[0] - self.swipe_start_pos[0]
        
        # 逆方向への移動が検出された場合リセット
        if total_dx < 0:
            self._reset_detection()
            return False
        
        # 最小距離に達したら検証段階へ
        if total_dx >= self.min_distance:
            self.state = SwipeState.VALIDATING
        
        return False
    
    def _validate_swipe(self) -> bool:
        """スワイプの検証"""
        current_time = self.time_buffer[-1]
        duration = current_time - self.swipe_start_time
        
        # 最小継続時間チェック
        if duration < self.min_duration:
            return False
        
        # 最大継続時間チェック
        if duration > self.max_duration:
            self._reset_detection()
            return False
        
        # 詳細な検証を実行
        if self._validate_swipe_characteristics():
            # クールダウンチェック
            if current_time - self.last_confirmed_swipe > self.swipe_cooldown:
                self.total_swipes_detected += 1
                self.last_confirmed_swipe = current_time
                self._reset_detection()
                return True
        
        self._reset_detection()
        return False
    
    def _validate_swipe_characteristics(self) -> bool:
        """スワイプの特徴を詳細に検証"""
        if len(self.position_buffer) < 5:
            return False
        
        positions = np.array(list(self.position_buffer))
        times = np.array(list(self.time_buffer))
        
        # 開始時刻からの位置のみを使用
        start_idx = 0
        for i, t in enumerate(times):
            if t >= self.swipe_start_time:
                start_idx = i
                break
        
        swipe_positions = positions[start_idx:]
        swipe_times = times[start_idx:]
        
        if len(swipe_positions) < 3:
            return False
        
        # 1. 総移動距離チェック
        total_dx = swipe_positions[-1][0] - swipe_positions[0][0]
        total_dy = swipe_positions[-1][1] - swipe_positions[0][1]
        
        if total_dx < self.min_distance:
            return False
        
        # 2. Y方向の偏差チェック
        y_deviation = abs(total_dy) / total_dx if total_dx > 0 else float('inf')
        if y_deviation > self.max_y_deviation:
            self.false_positives_filtered += 1
            return False
        
        # 3. 平均速度チェック
        duration = swipe_times[-1] - swipe_times[0]
        if duration <= 0:
            return False
        
        avg_velocity = total_dx / duration
        if avg_velocity < self.min_velocity or avg_velocity > self.max_velocity:
            self.false_positives_filtered += 1
            return False
        
        # 4. 一貫性チェック（右方向の移動が継続しているか）
        consistent_right_movement = True
        for i in range(1, len(swipe_positions)):
            dx = swipe_positions[i][0] - swipe_positions[i-1][0]
            # より多くの逆方向移動を許容（IRカメラのノイズ対策）
            if dx < -15:  # 15px以上の逆方向移動は不許可（より寛容に）
                consistent_right_movement = False
                break
        
        if not consistent_right_movement:
            self.false_positives_filtered += 1
            return False
        
        return True
    
    def _reset_detection(self):
        """検出状態をリセット"""
        self.state = SwipeState.IDLE
        self.swipe_start_time = None
        self.swipe_start_pos = None
    
    def get_current_swipe_progress(self) -> Optional[dict]:
        """現在のスワイプ進行状況を取得（デバッグ用）"""
        if self.state == SwipeState.IDLE or not self.swipe_start_pos or not self.position_buffer:
            return None
        
        current_pos = self.position_buffer[-1]
        current_time = self.time_buffer[-1]
        
        total_dx = current_pos[0] - self.swipe_start_pos[0]
        duration = current_time - self.swipe_start_time
        velocity = total_dx / duration if duration > 0 else 0
        
        return {
            'state': self.state.value,
            'distance': total_dx,
            'duration': duration,
            'velocity': velocity,
            'progress': min(total_dx / self.min_distance, 1.0)
        }
    
    def get_statistics(self) -> dict:
        """検出統計を取得"""
        return {
            'total_swipes_detected': self.total_swipes_detected,
            'false_positives_filtered': self.false_positives_filtered,
            'current_state': self.state.value
        }
    
    def reset_statistics(self):
        """統計をリセット"""
        self.total_swipes_detected = 0
        self.false_positives_filtered = 0