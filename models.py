import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Union

@dataclass
class VideoInfo:
    total_frames: int = 0
    fps: float = 30.0
    frame_width: int = 1280
    frame_height: int = 720
    target_width: int = 1280
    target_height: int = 720
    processed_frames: int = 0
    duration: float = 0.0

@dataclass
class PassEvent:
    time: float
    from_player: int
    to_player: int
    team: str
    from_position: Tuple[int, int]
    to_position: Tuple[int, int]
    completed: bool = True
    length: float = 0.0
    zone_from: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    zone_to: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    direction: str = "forward"  # forward, backward, lateral
    pass_type: str = "ground"   # ground, lofted, through
    progressive: bool = False   # Moves ball significantly forward
    danger_zone: bool = False   # Pass to final third or penalty area
    xA: float = 0.0            # Expected assists value
    breaking_lines: bool = False  # Pass that breaks defensive lines
    switch_play: bool = False     # Long cross-field pass

@dataclass
class ShotEvent:
    time: float
    player: int
    team: str
    position: Tuple[int, int]
    target_goal: str
    on_target: bool = False
    goal: bool = False
    expected_goal: float = 0.0
    distance: float = 0.0
    angle: float = 0.0
    shot_type: str = "normal"  # normal, volley, header, free_kick, penalty
    scenario: str = "open_play"  # open_play, set_piece, counter_attack
    pressure: float = 0.0  # A measure of defensive pressure during shot (0-1)
    zone: str = "central"  # central, left_side, right_side, box, outside_box
    
@dataclass
class Formation:
    shape: str  # e.g. "4-3-3"
    positions: List[Tuple[int, int]]
    confidence: float = 0.0
    timestamp: float = 0.0
    team: str = ""
    
    def to_dict(self):
        return {
            "shape": self.shape,
            "positions": self.positions,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "team": self.team
        }

@dataclass
class Detection:
    """Enhanced detection class for players and ball"""
    id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]  # (center_x, center_y)
    confidence: float
    class_id: int  # 0=player, 1=ball
    team: Optional[str] = None
    dominant_color: Optional[Tuple[int, int, int]] = None
    keypoints: Optional[List[Tuple[float, float, float]]] = None  # For pose estimation
    velocity: Optional[Tuple[float, float]] = None
    is_occluded: bool = False
    track_id: Optional[int] = None  # For consistent tracking across frames
    
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    def area(self) -> float:
        return self.width() * self.height()

@dataclass
class MatchConfig:
    """Match configuration settings"""
    team_home: str = "Home Team"
    team_away: str = "Away Team"
    home_color: str = "#0000FF"  # Blue
    away_color: str = "#FF0000"  # Red
    home_playstyle: str = "Possession-Based"
    away_playstyle: str = "Counter-Attack"
    confidence_threshold: float = 0.3
    tracking_memory: int = 50
    frame_skip: int = 3
    field_width: float = 105  # meters
    field_height: float = 68  # meters
    use_gpu: bool = True
    use_half_precision: bool = True
    batch_size: int = 4
    iou_threshold: float = 0.5
    analysis_resolution: str = "Medium (720p)"
    max_frames_to_process: int = 5000
    formation_detection_method: str = "Enhanced"
    pass_detection_sensitivity: int = 5
    shot_detection_sensitivity: int = 5

# Constants for field dimensions (standard soccer field in meters)
FIELD_WIDTH = 105  # meters
FIELD_HEIGHT = 68  # meters

# Session state keys
KEY_ANALYSIS_COMPLETE = "analysis_complete"
KEY_PLAYER_STATS = "player_stats"
KEY_TEAM_STATS = "team_stats"
KEY_EVENTS = "events"
KEY_VIDEO_INFO = "video_info"