import cv2
import torch
import numpy as np
import pandas as pd
import time
import os
import traceback
import tempfile
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import deque, defaultdict
from scipy.spatial import distance, ConvexHull
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN, KMeans
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import requests

from models import (
    VideoInfo, PassEvent, ShotEvent, Formation, Detection, MatchConfig,
    FIELD_WIDTH, FIELD_HEIGHT, KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS,
    KEY_TEAM_STATS, KEY_EVENTS, KEY_VIDEO_INFO
)

from tracking import EnhancedTracker

class FootballAnalyzer:
    def __init__(self, st):
        """Initialize the football analyzer with Streamlit context"""
        self.st = st
        
        # Initialize session state for persistence across reruns
        self.initialize_session_state()
        
        # Set up page title
        st.title("⚽ Advanced Football Match Analysis Platform")
        
        # Initialize sidebar
        self.setup_sidebar()
        
        # Set up tabs for different analyses
        self.setup_tabs()
        
        # Initialize configuration
        self.config = self.create_match_config()
        
        # Initialize video information with defaults
        self.video_info = VideoInfo()
        
        # Initialize data structures
        self.initialize_data_structures()
        
        # Initialize tactical engine
        self.initialize_tactical_engine()
        
        # Initialize tracker
        if self.check_model_settings_complete():
            try:
                self.initialize_tracker()
            except Exception as e:
                self.st.sidebar.error(f"Error initializing tracker: {str(e)}")
    
    def initialize_session_state(self):
        """Initialize session state variables for persistence across reruns"""
        if 'page' not in self.st.session_state:
            self.st.session_state.page = 'main'
            
        if KEY_ANALYSIS_COMPLETE not in self.st.session_state:
            self.st.session_state[KEY_ANALYSIS_COMPLETE] = False
            
        if KEY_PLAYER_STATS not in self.st.session_state:
            self.st.session_state[KEY_PLAYER_STATS] = None
            
        if KEY_TEAM_STATS not in self.st.session_state:
            self.st.session_state[KEY_TEAM_STATS] = None
            
        if KEY_EVENTS not in self.st.session_state:
            self.st.session_state[KEY_EVENTS] = None
            
        if KEY_VIDEO_INFO not in self.st.session_state:
            self.st.session_state[KEY_VIDEO_INFO] = VideoInfo()
            
        if 'processed_video_path' not in self.st.session_state:
            self.st.session_state.processed_video_path = None
            
        if 'gemini_api_key' not in self.st.session_state:
            self.st.session_state.gemini_api_key = None
    
    def create_match_config(self) -> MatchConfig:
        """Create match configuration from sidebar inputs"""
        config = MatchConfig(
            team_home=self.team_home,
            team_away=self.team_away,
            home_color=self.home_color,
            away_color=self.away_color,
            home_playstyle=self.home_playstyle,
            away_playstyle=self.away_playstyle,
            confidence_threshold=self.confidence_threshold,
            tracking_memory=self.tracking_memory,
            frame_skip=self.frame_skip,
            use_gpu=self.use_gpu,
            use_half_precision=self.use_half_precision,
            batch_size=self.batch_size,
            iou_threshold=self.iou_threshold,
            analysis_resolution=self.analysis_resolution,
            max_frames_to_process=self.max_frames_to_process,
            formation_detection_method=self.formation_detection_method,
            pass_detection_sensitivity=self.pass_detection_sensitivity,
            shot_detection_sensitivity=self.shot_detection_sensitivity
        )
        return config
    
    def setup_sidebar(self):
        """Set up the sidebar with input options"""
        self.st.sidebar.title("⚙️ Analysis Settings")
        
        # Team information
        self.team_home = self.st.sidebar.text_input("🏠 Home Team Name", "Home Team")
        self.team_away = self.st.sidebar.text_input("🚀 Away Team Name", "Away Team")
        
        self.home_color = self.st.sidebar.color_picker("🎽 Home Team Jersey Color", "#0000FF")
        self.away_color = self.st.sidebar.color_picker("🎽 Away Team Jersey Color", "#FF0000")
        
        # Style of play description
        self.st.sidebar.markdown("### Team Playing Styles")
        self.home_playstyle = self.st.sidebar.selectbox(
            "Home Team Style", 
            ["Possession-Based", "Counter-Attack", "High-Press", "Defensive", "Direct Play", "Custom"],
            index=0
        )
        if self.home_playstyle == "Custom":
            self.home_playstyle_custom = self.st.sidebar.text_area("Describe Home Team Style", "")
            
        self.away_playstyle = self.st.sidebar.selectbox(
            "Away Team Style", 
            ["Possession-Based", "Counter-Attack", "High-Press", "Defensive", "Direct Play", "Custom"],
            index=1
        )
        if self.away_playstyle == "Custom":
            self.away_playstyle_custom = self.st.sidebar.text_area("Describe Away Team Style", "")
        
        # Analysis settings
        self.confidence_threshold = self.st.sidebar.slider("Detection Confidence Threshold", 0.1, 0.9, 0.3)
        self.tracking_memory = self.st.sidebar.slider("Player Tracking Memory (frames)", 10, 200, 50)
        self.frame_skip = self.st.sidebar.slider("Process every N frames", 1, 10, 3)
        
        # Advanced settings with expander
        with self.st.sidebar.expander("Advanced Settings"):
            self.use_gpu = self.st.checkbox("Use GPU (if available)", True)
            self.use_half_precision = self.st.checkbox("Use Half Precision (FP16)", True)
            self.batch_size = self.st.slider("Batch Size", 1, 16, 4)
            self.iou_threshold = self.st.slider("IOU Threshold", 0.1, 0.9, 0.5)
            self.analysis_resolution = self.st.select_slider(
                "Analysis Resolution",
                options=["Low (360p)", "Medium (720p)", "High (1080p)"],
                value="Medium (720p)"
            )
            
            # Formation detection settings
            self.formation_detection_method = self.st.selectbox(
                "Formation Detection Method",
                ["Basic", "Clustering", "Enhanced"],
                index=2
            )
            
            # Pass detection sensitivity
            self.pass_detection_sensitivity = self.st.slider(
                "Pass Detection Sensitivity", 
                1, 10, 5,
                help="Higher values detect more passes but may include false positives"
            )
            
            # Enhanced shot detection sensitivity
            self.shot_detection_sensitivity = self.st.slider(
                "Shot Detection Sensitivity", 
                1, 10, 5,
                help="Higher values detect more shots but may include false positives"
            )
            
            # Limit frames for processing
            self.max_frames_to_process = self.st.slider(
                "Max Frames to Process", 
                500, 10000, 5000,
                help="Limit the number of frames to process for faster analysis"
            )
        
        # Model options
        with self.st.sidebar.expander("Model Settings"):
            self.model_source = self.st.radio(
                "Model Source",
                ["Default YOLOv8", "Upload Custom Model"],
                index=0
            )
            
            if self.model_source == "Upload Custom Model":
                self.custom_model_file = self.st.file_uploader("Upload Custom YOLO Model", type=["pt", "pth"])
                self.model_description = self.st.text_input("Model Description (optional)", "")
            else:
                self.custom_model_file = None
        
        # Video upload
        # Video input: upload or use local path
        self.video_input_mode = self.st.sidebar.radio("📽 Video Source", ["Upload Video", "Local File Path"], index=0)

        if self.video_input_mode == "Upload Video":
            self.video_path = self.st.sidebar.file_uploader("📂 Upload Match Video", type=["mp4", "avi", "mov"])
        else:
            self.local_video_path = self.st.sidebar.text_input("📁 Enter Local Video Path")
            if os.path.exists(self.local_video_path):
                self.video_path = self.local_video_path
            else:
                self.video_path = None
                self.st.warning("❗ Enter a valid path to a local video file.")

        # Analysis options
        self.enable_heatmap = self.st.sidebar.checkbox("Generate Heatmaps", True)
        self.enable_formation = self.st.sidebar.checkbox("Analyze Team Formations", True)
        self.enable_events = self.st.sidebar.checkbox("Detect Key Events", True)
        self.enable_report = self.st.sidebar.checkbox("Generate PDF Report", True)
        self.enable_tactical = self.st.sidebar.checkbox("Generate Tactical Analysis", True)
        
        # Gemini API integration
        with self.st.sidebar.expander("AI Integration"):
            self.enable_gemini = self.st.checkbox("Enable Gemini AI Insights", False)
            if self.enable_gemini:
                self.st.session_state.gemini_api_key = self.st.text_input(
                    "Gemini API Key", 
                    value=self.st.session_state.gemini_api_key if self.st.session_state.gemini_api_key else "",
                    type="password"
                )
                self.gemini_model = self.st.selectbox("Gemini Model", ["gemini-pro", "gemini-1.5-pro"], index=1)

                # ✅ Add this line to set the model instance
                import google.generativeai as genai
                genai.configure(api_key=self.st.session_state.gemini_api_key)
                self.gemini_model_instance = genai.GenerativeModel(self.gemini_model)


        # Start analysis button
        self.start_analysis = self.st.sidebar.button("🚀 Start Analysis")
        
        # Reset analysis button
        if self.st.session_state[KEY_ANALYSIS_COMPLETE]:
            if self.st.sidebar.button("🔄 Reset Analysis"):
                for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                    self.st.session_state[key] = None
                self.st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                self.st.session_state[KEY_ANALYSIS_COMPLETE] = False
                self.st.session_state.processed_video_path = None
                self.st.experimental_rerun()
    
    def setup_tabs(self):
        """Set up tabs for different analyses"""
        self.tab1, self.tab2, self.tab3, self.tab4, self.tab5, self.tab6, self.tab7 = self.st.tabs([
            "📹 Video Analysis", 
            "🔍 Player Stats", 
            "🌍 Spatial Analysis",
            "📊 Team Analysis",
            "💪 Strengths & Weaknesses",
            "🎯 Tactical Suggestions",
            "📝 Report"
        ])
    
    def check_model_settings_complete(self):
        """Check if model settings are complete and valid"""
        if self.model_source == "Upload Custom Model" and not self.custom_model_file:
            return False
        return True
    
    def initialize_tracker(self):
        """Initialize the enhanced tracker"""
        with self.st.spinner("Loading models..."):
            # Determine model path
            try:
                if self.model_source == "Upload Custom Model" and self.custom_model_file is not None:
                    # Save uploaded model file to temp location
                    model_bytes = self.custom_model_file.read()
                    temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
                    temp_model_file.write(model_bytes)
                    model_path = temp_model_file.name
                    temp_model_file.close()
                    
                    self.st.sidebar.success(f"✅ Custom model loaded: {self.custom_model_file.name}")
                else:
                    # Use default model path
                    model_path = "yolov8x-pose.pt"
                    if not os.path.exists(model_path):
                        model_path = "yolov8x.pt"  # Fallback to standard model
                
                # Initialize enhanced tracker
                self.tracker = EnhancedTracker(self.config, model_path)
                
                if hasattr(self.tracker, 'model') and self.tracker.model is not None:
                    self.st.sidebar.success(f"✅ Enhanced tracker initialized successfully")
                else:
                    self.st.sidebar.warning("Tracker initialized but model loading failed. Some features may be limited.")
            except Exception as e:
                self.st.error(f"Error initializing tracker: {str(e)}")
                traceback.print_exc()
                self.tracker = None
    
    def initialize_data_structures(self):
        """Initialize all data structures for tracking and analysis"""
        # Load video info from session state if available
        if self.st.session_state[KEY_VIDEO_INFO]:
            self.video_info = self.st.session_state[KEY_VIDEO_INFO]
        
        # Player tracking
        self.player_positions = defaultdict(lambda: deque(maxlen=self.tracking_memory))
        self.player_velocities = defaultdict(lambda: deque(maxlen=self.tracking_memory))
        self.player_team = {}  # Map player ID to team
        self.speed_data = defaultdict(list)
        self.distance_data = defaultdict(float)
        self.acceleration_data = defaultdict(list)
        self.ball_possession = defaultdict(int)
        self.ball_positions = deque(maxlen=self.tracking_memory)
        self.ball_velocities = deque(maxlen=self.tracking_memory)
        
        # Team analysis
        self.team_possession_frames = {self.team_home: 0, self.team_away: 0}
        self.team_positions = {self.team_home: [], self.team_away: []}
        self.team_formations = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        
        # Enhanced formation tracking
        self.formation_history = {self.team_home: [], self.team_away: []}
        self.formation_transitions = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        
        # Event detection
        self.events = []
        self.pass_data = []
        self.shot_data = []
        self.defensive_actions = defaultdict(list)
        self.pressing_data = defaultdict(list)
        
        # Enhanced pass analysis
        self.pass_success_rate = {self.team_home: 0, self.team_away: 0}
        self.pass_types = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.pass_directions = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.pass_networks = {self.team_home: defaultdict(lambda: defaultdict(int)), 
                             self.team_away: defaultdict(lambda: defaultdict(int))}
        self.progressive_passes = {self.team_home: 0, self.team_away: 0}
        self.danger_zone_passes = {self.team_home: 0, self.team_away: 0}
        self.pass_length_distribution = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.breaking_lines_passes = {self.team_home: 0, self.team_away: 0}
        self.switch_play_passes = {self.team_home: 0, self.team_away: 0}
        self.total_xA = {self.team_home: 0.0, self.team_away: 0.0}
        
        # Enhanced shot analysis
        self.shot_types = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.shot_scenarios = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.shot_zones = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.total_xG = {self.team_home: 0.0, self.team_away: 0.0}
        self.shots_under_pressure = {self.team_home: 0, self.team_away: 0}
        self.shot_success_rate = {self.team_home: 0.0, self.team_away: 0.0}
        self.shot_efficiency = {self.team_home: 0.0, self.team_away: 0.0}  # Goals vs xG
        
        # Pitch zones (divide pitch into a 6x9 grid)
        self.zone_possession = np.zeros((6, 9, 2))  # Last dimension: [home, away]
        self.zone_passes = np.zeros((6, 9, 2))
        self.zone_shots = np.zeros((6, 9, 2))
        self.zone_defensive_actions = np.zeros((6, 9, 2))
        self.zone_pressure = np.zeros((6, 9, 2))

        # Team strengths and weaknesses
        self.team_strengths = {self.team_home: {}, self.team_away: {}}
        self.team_weaknesses = {self.team_home: {}, self.team_away: {}}
        
        # Tactical analysis data
        self.pressing_intensity = {self.team_home: 0, self.team_away: 0}
        self.defensive_line_height = {self.team_home: 0, self.team_away: 0}
        self.pass_length_data = {self.team_home: [], self.team_away: []}
        self.buildup_patterns = {self.team_home: defaultdict(int), self.team_away: defaultdict(int)}
        self.tactical_suggestions = {self.team_home: [], self.team_away: []}
        
        # Individual player roles and performance
        self.player_roles = {}
        self.player_performance = {}
        
        # Analysis results
        self.analysis_results = {}
        
        # For tracking
        self.prev_positions = {}
        self.prev_frame_time = None
        
        # Gemini API integration
        self.gemini_insights = {self.team_home: [], self.team_away: []}
        
        # Initialize player_stats_df with an empty DataFrame to avoid errors
        self.player_stats_df = pd.DataFrame()
        
        # Initialize team_stats with default values
        self.team_stats = {
            self.team_home: {
                'Possession (%)': 0,
                'Distance (m)': 0,
                'Passes': 0,
                'Shots': 0
            },
            self.team_away: {
                'Possession (%)': 0,
                'Distance (m)': 0,
                'Passes': 0,
                'Shots': 0
            }
        }
    
    def initialize_tactical_engine(self):
        """Initialize the tactical analysis engine"""
        # Define tactical patterns for different play styles
        self.playstyle_patterns = {
            "Possession-Based": {
                "pass_length": "short",
                "pass_tempo": "high",
                "defensive_line": "high",
                "pressing_intensity": "medium",
                "width": "wide",
                "counter_attack_speed": "low",
                "key_zones": [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]
            },
            "Counter-Attack": {
                "pass_length": "long",
                "pass_tempo": "low",
                "defensive_line": "low",
                "pressing_intensity": "low",
                "width": "narrow",
                "counter_attack_speed": "high",
                "key_zones": [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)]
            },
            "High-Press": {
                "pass_length": "short",
                "pass_tempo": "high",
                "defensive_line": "high",
                "pressing_intensity": "high",
                "width": "wide",
                "counter_attack_speed": "medium",
                "key_zones": [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5)]
            },
            "Defensive": {
                "pass_length": "mixed",
                "pass_tempo": "low",
                "defensive_line": "low",
                "pressing_intensity": "low",
                "width": "narrow",
                "counter_attack_speed": "medium",
                "key_zones": [(4, 3), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
            },
            "Direct Play": {
                "pass_length": "long",
                "pass_tempo": "medium",
                "defensive_line": "medium",
                "pressing_intensity": "medium",
                "width": "wide",
                "counter_attack_speed": "high",
                "key_zones": [(2, 0), (2, 8), (3, 0), (3, 8)]
            }
        }
        
        # Counter-strategies for each play style
        self.counter_strategies = {
            "Possession-Based": [
                "Apply high pressing to disrupt build-up play",
                "Maintain compact defensive shape to limit passing options",
                "Quick transitions when ball is won to exploit space behind high defensive line",
                "Target pressing triggers (back passes, slow sideways passes)",
                "Overload central areas to force play wide"
            ],
            "Counter-Attack": [
                "Maintain possession to limit counter-attacking opportunities",
                "Position players to prevent long clearances and transitions",
                "Use pressing traps in opponent's half",
                "Maintain defensive awareness even in attacking phases",
                "Apply quick counter-pressing when possession is lost"
            ],
            "High-Press": [
                "Use direct passes to bypass the press",
                "Position tall players for long balls",
                "Quick ball movement with one-touch passing",
                "Use goalkeeper as additional passing option",
                "Create numerical advantage in build-up phase with dropping midfielders"
            ],
            "Defensive": [
                "Patient build-up to draw out defensive block",
                "Use width to stretch defensive structure",
                "Quick switches of play to create space",
                "Utilize creative players between defensive lines",
                "Use set pieces effectively as scoring opportunities"
            ],
            "Direct Play": [
                "Maintain strong aerial presence in defense",
                "Position for second balls after long passes",
                "Apply pressure on wide areas to prevent crosses",
                "Maintain compact defensive shape vertically",
                "Use technical players to retain possession when winning the ball"
            ],
            "Custom": [
                "Analyze opponent patterns throughout the match",
                "Focus on exploiting spaces when opposition changes formation",
                "Adjust pressing strategy based on opponent build-up patterns",
                "Target transitions during opponent's attacking phases",
                "Adapt formation to counter opponent's key playmakers"
            ]
        }
        
        # Tactical weakness indicators
        self.tactical_weaknesses = {
            "possession_loss_own_half": "Vulnerable to high pressing",
            "low_possession_percentage": "Difficulty maintaining control",
            "low_pass_completion": "Inconsistent build-up play",
            "high_goals_conceded_counter": "Vulnerable to counter-attacks",
            "low_defensive_duels_won": "Weak in defensive duels",
            "high_crosses_conceded": "Vulnerable in wide areas",
            "low_aerial_duels_won": "Weak in aerial situations",
            "high_shots_conceded_box": "Poor box defense",
            "low_shots_on_target": "Inefficient attacking",
            "low_pressing_success": "Ineffective pressing system",
            "low_forward_passes": "Lacks attacking progression",
            "high_lateral_passes": "Predictable sideways passing",
            "low_progressive_passes": "Difficulty progressing up the field",
            "poor_defensive_transitions": "Vulnerable during defensive transitions",
            "low_xG_per_shot": "Poor shot quality selection"
        }
        
        # Tactical strength indicators
        self.tactical_strengths = {
            "high_possession_percentage": "Strong ball retention",
            "high_pass_completion": "Effective build-up play",
            "high_passes_final_third": "Creative in attack",
            "high_crosses_completed": "Effective wide play",
            "high_pressing_success": "Effective pressing system",
            "high_defensive_duels_won": "Strong in defensive duels",
            "high_aerial_duels_won": "Strong in aerial situations",
            "low_shots_conceded_box": "Solid box defense",
            "high_shots_on_target": "Efficient attacking",
            "high_counter_attacks": "Effective on transitions",
            "high_forward_passes": "Progressive attacking play",
            "high_through_balls": "Creative penetrative passing",
            "high_progressive_passes": "Excellent at advancing the ball",
            "strong_defensive_transitions": "Quick recovery after losing possession",
            "high_xG_per_shot": "Creates high-quality chances"
        }
        
        # Initialize with default values for pressing intensity
        self.pressing_intensity = {self.team_home: 60, self.team_away: 60}
    def generate_llm_report_summary(self):
        if not self.enable_gemini or not self.st.session_state.gemini_api_key:
            return "Gemini AI integration is disabled or API key is missing."

        try:
            model = self.gemini_model_instance  # Already initialized in your existing Gemini logic

            prompt = f"""
            You are an elite football analyst. Write a comprehensive match analysis report.

            Match: {self.team_home} vs {self.team_away}

            Team Stats:
            {json.dumps(self.team_stats, indent=2)}

            Top Player Stats:
            {self.player_stats_df.head(5).to_dict(orient='records')}

            Write this report in formal, clear language. Include:
            - Tactical observations
            - Strengths and weaknesses
            - Key player performances
            - Suggestions for future improvement
            """

            response = model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"(AI summary generation failed: {str(e)})"

    def preprocess_video(self, video_file):
        """Preprocess uploaded or local video file and return video capture"""
        try:
            # CASE 1: Local file path (string)
            if isinstance(video_file, str) and os.path.exists(video_file):
                video_path = video_file

            # CASE 2: Uploaded file (Streamlit UploadedFile)
            elif video_file is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(video_file.read())
                video_path = temp_file.name
                temp_file.close()
            else:
                self.st.error("No valid video file provided.")
                return None, None

            # Load video properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.st.error("Error: Could not open video file.")
                return None, None

            # Extract and store video info
            self.video_info.fps = cap.get(cv2.CAP_PROP_FPS)
            self.video_info.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_info.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_info.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_info.duration = self.video_info.total_frames / self.video_info.fps

            # Resize based on resolution
            if self.analysis_resolution == "Low (360p)":
                self.video_info.target_height = 360
            elif self.analysis_resolution == "Medium (720p)":
                self.video_info.target_height = 720
            else:
                self.video_info.target_height = 1080

            scale = self.video_info.target_height / self.video_info.frame_height
            self.video_info.target_width = int(self.video_info.frame_width * scale)

            self.st.session_state[KEY_VIDEO_INFO] = self.video_info
            return cap, video_path

        except Exception as e:
            self.st.error(f"Error processing video file: {str(e)}")
            traceback.print_exc()
            return None, None

    
    def process_video(self):
        """Process the video with enhanced player detection and tracking"""
        try:
            # Make sure tracker is initialized
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.initialize_tracker()
                
            # Safety check
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.st.error("Tracker initialization failed")
                return None
                
            cap, temp_video_path = self.preprocess_video(self.video_path)
            if cap is None:
                self.st.error("Error loading video file.")
                return None
            
            # Create output video writer
            output_video_path = "enhanced_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.video_info.fps, 
                                 (self.video_info.target_width, self.video_info.target_height))
            
            # Set up progress bar
            progress_bar = self.st.progress(0)
            status_text = self.st.empty()
            
            # Process frames
            frames_to_process = min(self.video_info.total_frames, self.max_frames_to_process)
            frame_count = 0
            processed = 0
            
            # Initialize time delta for tracking
            dt = 1.0 / self.video_info.fps * self.frame_skip
            
            while cap.isOpened() and frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames for faster processing
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Resize frame for analysis
                frame_resized = cv2.resize(frame, (self.video_info.target_width, self.video_info.target_height))
                
                # Use enhanced tracker for detection and tracking
                current_time = frame_count / self.video_info.fps
                try:
                    detections = self.tracker.detect_and_track(frame_resized, frame_count, dt)
                    
                    # Process detections to update statistics
                    self.process_detections(detections, frame_count)
                    
                    # Draw enhanced visualizations
                    processed_frame = self.tracker.draw_enhanced_detections(frame_resized, detections)
                    
                    # Analyze formations if enabled
                    if self.enable_formation and len(detections) > 0:
                        player_detections = [d for d in detections if d.class_id == 0]
                        self.analyze_formations(player_detections, processed_frame, frame_count)
                    
                    # Detect events if enabled
                    if self.enable_events:
                        ball_position = None
                        for d in detections:
                            if d.class_id == 1:  # Ball
                                ball_position = d.center
                                break
                        
                        if ball_position:
                            self.detect_events(frame_count, [d for d in detections if d.class_id == 0], ball_position)
                    
                    # Display frame time
                    cv2.putText(processed_frame, f"Time: {current_time:.2f}s", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Write processed frame to output video
                    out.write(processed_frame)
                except Exception as e:
                    # On error, log but continue with original frame
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    cv2.putText(frame_resized, f"Error: {str(e)[:50]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    out.write(frame_resized)
                
                # Update progress
                processed += 1
                progress = min(processed / (frames_to_process / self.frame_skip), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{frames_to_process} ({progress*100:.1f}%)")
                
                frame_count += 1
                self.video_info.processed_frames += 1
                
            # Release resources
            cap.release()
            out.release()
            
            # Clean up temp file
            try:
                os.unlink(temp_video_path)
            except:
                pass
            
            # Save video path to session state
            self.st.session_state.processed_video_path = output_video_path
            
            # Update video info in session state
            self.st.session_state[KEY_VIDEO_INFO] = self.video_info
            
            # Prepare analysis results
            self.prepare_analysis_results()
            
            # Generate AI insights if enabled
            if self.enable_gemini and self.st.session_state.gemini_api_key:
                self.generate_gemini_insights()
            
            # Analyze strengths and weaknesses
            self.analyze_strengths_weaknesses()
            
            # Set analysis complete flag in session state
            self.st.session_state[KEY_ANALYSIS_COMPLETE] = True
            
            # Return path to processed video
            return output_video_path
        except Exception as e:
            self.st.error(f"Error processing video: {str(e)}")
            traceback.print_exc()
            return None
    
    def display_video_analysis(self, output_video_path):
        """Display processed video and basic analysis"""
        try:
            with self.tab1:
                self.st.subheader("📹 Processed Video with Player Tracking")
                if os.path.exists(output_video_path):
                    self.st.video(output_video_path)
                else:
                    self.st.error("Video file not found. The analysis may have failed.")
                    return
                
                # Display basic match info
                col1, col2 = self.st.columns(2)
                
                with col1:
                    self.st.subheader(f"⚔️ {self.team_home} vs {self.team_away}")
                    self.st.write(f"Total Frames Analyzed: {self.video_info.processed_frames}")
                    self.st.write(f"Video Length: {self.video_info.duration:.2f} seconds")
                    if hasattr(self, 'player_positions'):
                        self.st.write(f"Analyzed Players: {len(self.player_positions)}")
                    else:
                        self.st.write("No player data available")
                
                with col2:
                    # Display possession pie chart
                    if sum(self.team_possession_frames.values()) > 0:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        labels = [self.team_home, self.team_away]
                        sizes = [self.team_stats[self.team_home]['Possession (%)'], 
                                self.team_stats[self.team_away]['Possession (%)']]
                        colors = [self.home_color, self.away_color]
                        
                        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        plt.title('Ball Possession')
                        self.st.pyplot(fig)
                    else:
                        self.st.info("Possession data not available")
                
                # Display events timeline
                if self.events:
                    self.st.subheader("⏱️ Key Events Timeline")
                    
                    # Group events into expandable sections by 15-minute intervals
                    events_by_interval = {}
                    interval_size = 15 * 60  # 15 minutes in seconds
                    
                    for event in sorted(self.events, key=lambda e: e['time']):
                        interval = int(event['time'] // interval_size)
                        interval_str = f"{interval * 15}-{(interval + 1) * 15} min"
                        
                        if interval_str not in events_by_interval:
                            events_by_interval[interval_str] = []
                        
                        events_by_interval[interval_str].append(event)
                    
                    # Display events by interval in expandable sections
                    for interval_str, interval_events in events_by_interval.items():
                        with self.st.expander(f"Events ({interval_str})"):
                            for event in interval_events:
                                time_str = f"{int(event['time'] // 60)}:{int(event['time'] % 60):02d}"
                                team = event['team']
                                team_color = self.home_color if team == self.team_home else self.away_color
                                
                                if event['type'] == 'pass':
                                    # Enhanced pass description
                                    direction = event.get('direction', 'unknown')
                                    pass_type = event.get('pass_type', 'normal')
                                    progressive = "progressive " if event.get('progressive', False) else ""
                                    danger_zone = "into danger zone " if event.get('danger_zone', False) else ""
                                    breaking_lines = "breaking lines " if event.get('breaking_lines', False) else ""
                                    switch_play = "switch play " if event.get('switch_play', False) else ""
                                    
                                    pass_quality = ""
                                    if event.get('xA', 0) > 0.1:
                                        pass_quality = "high-quality "
                                    
                                    self.st.markdown(f"<span style='color:{team_color}'>⏱️ {time_str} - **{pass_quality}{progressive}{breaking_lines}{danger_zone}{switch_play}{pass_type} {direction} pass** from Player {event['from_player']} to Player {event['to_player']} ({team})</span>", unsafe_allow_html=True)
                                
                                elif event['type'] == 'shot':
                                    # Enhanced shot description
                                    on_target = "on target" if event.get('on_target', False) else "off target"
                                    xg = f" (xG: {event.get('xG', 0):.2f})" if 'xG' in event else ""
                                    shot_type = event.get('shot_type', 'normal')
                                    zone = event.get('zone', '')
                                    pressure = "under pressure " if event.get('pressure', 0) > 0.5 else ""
                                    distance = f"from {event.get('distance', 0):.1f}m " if 'distance' in event else ""
                                    
                                    self.st.markdown(f"<span style='color:{team_color}'>⏱️ {time_str} - **{shot_type.capitalize()} shot {on_target}** {pressure}{distance}by Player {event['player']} ({team}) at {event['target_goal']} goal from {zone}{xg}</span>", unsafe_allow_html=True)
                else:
                    self.st.info("No events detected in the analysis")
        except Exception as e:
            self.st.error(f"Error displaying video analysis: {str(e)}")
            traceback.print_exc()
    
    def process_detections(self, detections, frame_idx):
        """Compatibility wrapper for update_statistics"""
        return self.update_statistics(detections, frame_idx)
        
    def update_statistics(self, detections, frame_idx):
        """Process detections to update statistics and tracking data"""
        try:
            current_time = frame_idx / self.video_info.fps
            
            # Get ball position
            ball_detection = None
            for detection in detections:
                if detection.class_id == 1:  # Ball class
                    ball_detection = detection
                    ball_position = detection.center
                    self.ball_positions.append(ball_position)
                    
                    if detection.velocity is not None:
                        self.ball_velocities.append(detection.velocity)
                    break
            
            # Process player detections
            player_detections = [d for d in detections if d.class_id == 0]
            
            for detection in player_detections:
                player_id = detection.track_id
                team = detection.team
                center = detection.center
                
                # Update player positions and team assignment
                self.player_positions[player_id].append(center)
                self.player_team[player_id] = team
                
                # Calculate distance moved since last frame
                if player_id in self.prev_positions:
                    prev_x, prev_y = self.prev_positions[player_id]
                    dx = center[0] - prev_x
                    dy = center[1] - prev_y
                    
                    # Calculate distance in pixels, then convert to meters using field ratio
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    
                    # Convert to real-world distance using field dimensions
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    distance_meters = distance_pixels * pixel_to_meter
                    
                    # Calculate speed (m/s) and acceleration
                    dt = 1.0 / self.video_info.fps * self.frame_skip
                    speed = distance_meters / dt
                    
                    # Calculate acceleration if we have previous speed data
                    if player_id in self.speed_data and len(self.speed_data[player_id]) > 0:
                        prev_speed = self.speed_data[player_id][-1]
                        acceleration = (speed - prev_speed) / dt
                        self.acceleration_data[player_id].append(acceleration)
                    
                    # Update tracking data
                    self.distance_data[player_id] += distance_meters
                    self.speed_data[player_id].append(speed)
                    
                    # Track position in zone grid for zone analysis
                    zone_x = min(int(center[0] / self.video_info.target_width * 9), 8)
                    zone_y = min(int(center[1] / self.video_info.target_height * 6), 5)
                    team_idx = 0 if team == self.team_home else 1
                    self.zone_possession[zone_y, zone_x, team_idx] += 1
                
                # Store current position for next frame
                self.prev_positions[player_id] = center
                
                # Update velocities
                if detection.velocity is not None:
                    self.player_velocities[player_id].append(detection.velocity)
            
            # Determine ball possession
            if ball_detection and player_detections:
                # Find closest player to ball
                nearest_player = min(player_detections, key=lambda p: 
                                    distance.euclidean(p.center, ball_detection.center))
                
                # Update ball possession
                self.ball_possession[nearest_player.track_id] += 1
                team = nearest_player.team
                self.team_possession_frames[team] += 1
                
        except Exception as e:
            print(f"Error updating statistics: {str(e)}")
            traceback.print_exc()
            
    def display_player_stats(self):
        """Display detailed player statistics"""
        try:
            with self.tab2:
                self.st.subheader("🔍 Player Statistics")
                
                # Check if player stats are available
                if not hasattr(self, 'player_stats_df') or self.player_stats_df.empty:
                    self.st.info("No player statistics available. Run the analysis first.")
                    return
                
                # Filter options
                team_filter = self.st.radio("Filter by Team", ["All", self.team_home, self.team_away], horizontal=True)
                
                # Apply filters
                if team_filter != "All":
                    filtered_df = self.player_stats_df[self.player_stats_df['Team'] == team_filter]
                else:
                    filtered_df = self.player_stats_df
                
                # Sort options
                sort_options = ["Distance (m)", "Avg Speed (m/s)", "Max Speed (m/s)", 
                                "Possession (%)", "Passes", "Passes Received", 
                                "Pass Completion (%)", "Progressive Passes", "Breaking Lines Passes",
                                "Expected Assists (xA)", "Shots", "Shots on Target", 
                                "Expected Goals (xG)", "Influence"]
                
                # Make sure sort_by is a column in the DataFrame
                available_columns = filtered_df.columns.tolist()
                sort_options = [col for col in sort_options if col in available_columns]
                
                sort_by = self.st.selectbox("Sort by", sort_options) if sort_options else "Distance (m)"
                
                # Display sorted table
                if sort_by in filtered_df.columns:
                    self.st.dataframe(filtered_df.sort_values(by=sort_by, ascending=False))
                else:
                    self.st.dataframe(filtered_df)
                
                # Display player distance comparison
                self.st.subheader("🏃 Player Distance Comparison")
                
                if 'Distance (m)' in filtered_df.columns and not filtered_df.empty:
                    fig = px.bar(filtered_df.sort_values(by="Distance (m)", ascending=False).head(10),
                                x="Player ID", y="Distance (m)", color="Team",
                                color_discrete_map={self.team_home: self.home_color, self.team_away: self.away_color})
                    
                    fig.update_layout(title="Top 10 Players by Distance Covered")
                    self.st.plotly_chart(fig)
                else:
                    self.st.info("Distance data not available.")
        except Exception as e:
            self.st.error(f"Error displaying player stats: {str(e)}")
            traceback.print_exc()
        """Process detections to update statistics and tracking data"""
        frame_idx = self.st.session_state.get("frame_idx", 0)
        current_time = frame_idx / self.video_info.fps

        try:
            current_time = frame_idx / self.video_info.fps
            
            # Get ball position
            ball_detection = None
            for detection in detections:
                if detection.class_id == 1:  # Ball class
                    ball_detection = detection
                    ball_position = detection.center
                    self.ball_positions.append(ball_position)
                    
                    if detection.velocity is not None:
                        self.ball_velocities.append(detection.velocity)
                    break
            detections = self.st.session_state.get("current_detections", [])

            # Process player detections
            player_detections = [d for d in detections if d.class_id == 0]
            
            for detection in player_detections:
                player_id = detection.track_id
                team = detection.team
                center = detection.center
                
                # Update player positions and team assignment
                self.player_positions[player_id].append(center)
                self.player_team[player_id] = team
                
                # Calculate distance moved since last frame
                if player_id in self.prev_positions:
                    prev_x, prev_y = self.prev_positions[player_id]
                    dx = center[0] - prev_x
                    dy = center[1] - prev_y
                    
                    # Calculate distance in pixels, then convert to meters using field ratio
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    
                    # Convert to real-world distance using field dimensions
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    distance_meters = distance_pixels * pixel_to_meter
                    
                    # Calculate speed (m/s) and acceleration
                    dt = 1.0 / self.video_info.fps * self.frame_skip
                    speed = distance_meters / dt
                    
                    # Calculate acceleration if we have previous speed data
                    if player_id in self.speed_data and len(self.speed_data[player_id]) > 0:
                        prev_speed = self.speed_data[player_id][-1]
                        acceleration = (speed - prev_speed) / dt
                        self.acceleration_data[player_id].append(acceleration)
                    
                    # Update tracking data
                    self.distance_data[player_id] += distance_meters
                    self.speed_data[player_id].append(speed)
                    
                    # Track position in zone grid for zone analysis
                    zone_x = min(int(center[0] / self.video_info.target_width * 9), 8)
                    zone_y = min(int(center[1] / self.video_info.target_height * 6), 5)
                    team_idx = 0 if team == self.team_home else 1
                    self.zone_possession[zone_y, zone_x, team_idx] += 1
                
                # Store current position for next frame
                self.prev_positions[player_id] = center
                
                # Update velocities
                if detection.velocity is not None:
                    self.player_velocities[player_id].append(detection.velocity)
            
            # Determine ball possession
            if ball_detection and player_detections:
                # Find closest player to ball
                nearest_player = min(player_detections, key=lambda p: 
                                    distance.euclidean(p.center, ball_detection.center))
                
                # Update ball possession
                self.ball_possession[nearest_player.track_id] += 1
                team = nearest_player.team
                self.team_possession_frames[team] += 1
                
        except Exception as e:
            print(f"Error updating statistics: {str(e)}")
            traceback.print_exc()
    
    def display_spatial_analysis(self):
        """Display spatial analysis like heatmaps and player movement"""
        try:
            with self.tab3:
                self.st.subheader("🌍 Spatial Analysis")
                
                # Player movement heatmap
                self.st.subheader("🔥 Player Movement Heatmaps")
                
                col1, col2 = self.st.columns(2)
                
                with col1:
                    self.st.markdown(f"#### {self.team_home} Team Heatmap")
                    self.generate_team_heatmap(self.team_home)
                
                with col2:
                    self.st.markdown(f"#### {self.team_away} Team Heatmap")
                    self.generate_team_heatmap(self.team_away)
        except Exception as e:
            self.st.error(f"Error displaying spatial analysis: {str(e)}")
            traceback.print_exc()
            
    def generate_team_heatmap(self, team):
        """Generate and display heatmap for a specific team"""
        try:
            # Collect all positions for this team
            positions = []
            for player_id, team_name in self.player_team.items():
                if team_name == team and player_id in self.player_positions:
                    positions.extend(self.player_positions[player_id])
            
            if positions:
                # Convert to numpy array
                positions = np.array(positions)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Draw field background
                rect = plt.Rectangle((0, 0), self.video_info.target_width, self.video_info.target_height, 
                                    facecolor='#238823', alpha=0.3, edgecolor='white')
                ax.add_patch(rect)
                
                # Draw center line and circle
                ax.plot([self.video_info.target_width/2, self.video_info.target_width/2], [0, self.video_info.target_height], 'white')
                center_circle = plt.Circle((self.video_info.target_width/2, self.video_info.target_height/2), 
                                         self.video_info.target_height/10, fill=False, color='white')
                ax.add_patch(center_circle)
                
                # Generate KDE for heatmap
                x = positions[:, 0]
                y = positions[:, 1]
                
                # Create 2D histogram
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=40, 
                                                      range=[[0, self.video_info.target_width], [0, self.video_info.target_height]])
                
                # Smooth the heatmap
                heatmap = gaussian_filter(heatmap, sigma=1.5)
                
                # Plot heatmap
                c = ax.imshow(heatmap.T, cmap='hot', origin='lower', 
                             extent=[0, self.video_info.target_width, 0, self.video_info.target_height],
                             alpha=0.7, interpolation='bilinear')
                
                # Remove axes
                ax.axis('off')
                
                # Add colorbar
                fig.colorbar(c, ax=ax, label='Presence Intensity')
                
                # Display
                self.st.pyplot(fig)
            else:
                self.st.info("Not enough data for heatmap generation.")
        except Exception as e:
            self.st.error(f"Error generating heatmap: {str(e)}")
            traceback.print_exc()
    
    def display_team_analysis(self):
        """Display team level analysis and comparisons"""
        try:
            with self.tab4:
                self.st.subheader("📊 Team Analysis")
                
                # Check if team stats are available
                if not self.team_stats:
                    self.st.info("No team statistics available. Run the analysis first.")
                    return
                
                # Create team comparison dataframe
                metrics = ['Possession (%)', 'Distance (m)', 'Passes', 'Pass Completion (%)', 
                          'Forward Passes (%)', 'Progressive Passes', 'Breaking Lines Passes',
                          'Expected Assists (xA)', 'Shots', 'Shots on Target', 'Shot Accuracy (%)',
                          'Expected Goals (xG)', 'Most Used Formation']
                
                # Make sure all metrics exist in team_stats
                available_metrics = []
                for m in metrics:
                    if m in self.team_stats[self.team_home] and m in self.team_stats[self.team_away]:
                        available_metrics.append(m)
                
                if available_metrics:
                    team_df = pd.DataFrame({
                        'Metric': available_metrics,
                        self.team_home: [self.team_stats[self.team_home].get(m, 'N/A') for m in available_metrics],
                        self.team_away: [self.team_stats[self.team_away].get(m, 'N/A') for m in available_metrics]
                    })
                    
                    # Display team comparison
                    self.st.subheader("⚔️ Team Comparison")
                    self.st.dataframe(team_df.set_index('Metric'))
                else:
                    self.st.warning("Team statistics are incomplete or not available.")
        except Exception as e:
            self.st.error(f"Error displaying team analysis: {str(e)}")
            traceback.print_exc()
    
    def display_strengths_weaknesses(self):
        """Display team strengths and weaknesses"""
        try:
            with self.tab5:
                self.st.subheader("💪 Team Strengths & Weaknesses Analysis")
                
                # Check if strengths and weaknesses are available
                if not self.team_strengths[self.team_home] and not self.team_strengths[self.team_away]:
                    self.st.info("Strengths and weaknesses analysis not available. Run the analysis first.")
                    return
                
                col1, col2 = self.st.columns(2)
                
                # Home team analysis
                with col1:
                    self.st.markdown(f"### {self.team_home}")
                    
                    # Strengths
                    self.st.markdown("#### 💪 Strengths")
                    if self.team_strengths[self.team_home]:
                        for key, data in self.team_strengths[self.team_home].items():
                            self.st.markdown(f"**{data['description']}** _{key}_")
                            # Create progress bar
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            self.st.progress(progress_value)
                    else:
                        self.st.info("No significant strengths identified.")
                    
                    # Weaknesses
                    self.st.markdown("#### 🔍 Areas for Improvement")
                    if self.team_weaknesses[self.team_home]:
                        for key, data in self.team_weaknesses[self.team_home].items():
                            self.st.markdown(f"**{data['description']}** _{key}_")
                            # Create inverted progress bar for weaknesses
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            self.st.progress(progress_value)
                    else:
                        self.st.info("No significant weaknesses identified.")
                
                # Away team analysis
                with col2:
                    self.st.markdown(f"### {self.team_away}")
                    
                    # Strengths
                    self.st.markdown("#### 💪 Strengths")
                    if self.team_strengths[self.team_away]:
                        for key, data in self.team_strengths[self.team_away].items():
                            self.st.markdown(f"**{data['description']}** _{key}_")
                            # Create progress bar
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            self.st.progress(progress_value)
                    else:
                        self.st.info("No significant strengths identified.")
                    
                    # Weaknesses
                    self.st.markdown("#### 🔍 Areas for Improvement")
                    if self.team_weaknesses[self.team_away]:
                        for key, data in self.team_weaknesses[self.team_away].items():
                            self.st.markdown(f"**{data['description']}** _{key}_")
                            # Create inverted progress bar for weaknesses
                            progress_value = min(max(data['value'], 0.1) / 100, 1.0)  # Ensure non-zero
                            self.st.progress(progress_value)
                    else:
                        self.st.info("No significant weaknesses identified.")
        except Exception as e:
            self.st.error(f"Error displaying strengths and weaknesses: {str(e)}")
            traceback.print_exc()
    
    def display_tactical_suggestions(self):
        """Display tactical suggestions based on analysis"""
        try:
            with self.tab6:
                self.st.subheader("🎯 Tactical Analysis & Suggestions")
                
                # Check if tactical suggestions are available
                if not self.tactical_suggestions[self.team_home] and not self.tactical_suggestions[self.team_away]:
                    self.st.info("Tactical suggestions not available. Run the analysis with tactical analysis enabled.")
                    return
                
                # Let user select which team to view suggestions for
                selected_team = self.st.radio(
                    "Select Team for Tactical Analysis",
                    [self.team_home, self.team_away],
                    horizontal=True
                )
                
                opponent = self.team_away if selected_team == self.team_home else self.team_home
                
                # Display opponent analysis
                self.st.subheader(f"Opponent Analysis: {opponent}")
                
                # Display opponent's playing style
                opponent_style = self.away_playstyle if opponent == self.team_away else self.home_playstyle
                
                # Display style characteristics
                if opponent_style in self.playstyle_patterns:
                    style_data = self.playstyle_patterns[opponent_style]
                    
                    col1, col2, col3 = self.st.columns(3)
                    
                    with col1:
                        self.st.metric("Playing Style", opponent_style)
                        self.st.metric("Pass Length", style_data["pass_length"].capitalize())
                    
                    with col2:
                        self.st.metric("Defensive Line", style_data["defensive_line"].capitalize())
                        self.st.metric("Pass Tempo", style_data["pass_tempo"].capitalize())
                    
                    with col3:
                        self.st.metric("Pressing Intensity", style_data["pressing_intensity"].capitalize())
                        self.st.metric("Width", style_data["width"].capitalize())
                elif opponent_style == "Custom":
                    self.st.info(f"Custom playing style for {opponent}")
                    custom_attr = f"{opponent.lower().replace(' ', '_')}_playstyle_custom"
                    if hasattr(self, custom_attr):
                        custom_style = getattr(self, custom_attr)
                        self.st.write(f"Description: {custom_style}")
                
                # Display tactical suggestions
                self.st.subheader("🎯 Tactical Suggestions")
                
                if self.tactical_suggestions.get(selected_team):
                    for i, suggestion in enumerate(self.tactical_suggestions[selected_team]):
                        # Create a card-like display for each suggestion
                        self.st.markdown(f"""
                        <div style="padding: 15px; border-left: 5px solid {self.home_color if selected_team == self.team_home else self.away_color}; 
                                background-color: rgba(0,0,0,0.03); margin-bottom: 15px; border-radius: 0 5px 5px 0;">
                            <h4>Strategy {i+1}</h4>
                            <p>{suggestion}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    self.st.info("No tactical suggestions available for this team.")
        except Exception as e:
            self.st.error(f"Error displaying tactical suggestions: {str(e)}")
            traceback.print_exc()
    
    def generate_report(self):
        """Generate a comprehensive PDF report with analysis results"""
        try:
            from fpdf import FPDF

            with self.tab7:
                self.st.subheader("📝 Match Analysis Report")

                if not self.st.session_state.get(KEY_ANALYSIS_COMPLETE, False):
                    self.st.warning("⚠️ Please run the match analysis first.")
                    return

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="Football Match Report", ln=True, align='C')
                pdf.ln(10)

                # Cover Page Info
                pdf.set_font("Arial", '', 12)
                pdf.cell(200, 10, txt=f"Match: {self.team_home} vs {self.team_away}", ln=True)
                pdf.cell(200, 10, txt=f"Total Duration: {self.video_info.duration:.2f} seconds", ln=True)
                pdf.ln(10)

                # Team Statistics
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Team Statistics", ln=True)
                pdf.set_font("Arial", '', 12)
                for team, stats in self.team_stats.items():
                    pdf.ln(5)
                    pdf.cell(200, 10, txt=f"{team}:", ln=True)
                    for k, v in stats.items():
                        pdf.cell(200, 8, txt=f"  - {k}: {v}", ln=True)
                    pdf.ln(2)

                # Top Player Stats
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt="Top 5 Players", ln=True)
                pdf.set_font("Arial", '', 12)
                for player_id, row in self.player_stats_df.head(5).iterrows():
                    pdf.cell(200, 10, txt=f"Player {player_id}:", ln=True)
                    for k, v in row.items():
                        pdf.cell(200, 8, txt=f"  - {k}: {v}", ln=True)
                    pdf.ln(2)

                # Include Gemini tactical analysis
                if self.enable_gemini and self.st.session_state.gemini_api_key:
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="AI-Powered Tactical Report", ln=True)
                    pdf.set_font("Arial", '', 12)

                    ai_summary = self.generate_llm_match_report()
                    clean_text = ai_summary.encode('latin-1', 'ignore').decode('latin-1')

                    for line in clean_text.split('\n'):
                        pdf.multi_cell(0, 8, line.strip())

                # Output PDF
                report_path = "match_report.pdf"
                pdf.output(report_path)

                # Download button
                with open(report_path, "rb") as f:
                    self.st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name="match_report.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            self.st.error(f"Error generating report: {str(e)}")
            traceback.print_exc()

    def analyze_formations(self, player_detections, frame, frame_idx):
        """Analyze and visualize team formations with enhanced algorithms"""
        try:
            # Group players by team
            home_players = [p for p in player_detections if p.team == self.team_home]
            away_players = [p for p in player_detections if p.team == self.team_away]
            
            # Process home team formation
            if len(home_players) > 5:  # Need at least a few players to analyze formation
                self.team_positions[self.team_home].append([p.center for p in home_players])
                
                # Determine formation using the selected method
                if self.formation_detection_method == "Basic":
                    formation, confidence = self.calculate_formation_basic(home_players)
                elif self.formation_detection_method == "Clustering":
                    formation, confidence = self.calculate_formation_clustering(home_players)
                else:  # Enhanced method
                    formation, confidence, positions = self.calculate_formation_enhanced(home_players)
                    
                    # Store formation history with timestamp
                    formation_data = Formation(
                        shape=formation,
                        positions=positions,
                        confidence=confidence,
                        timestamp=frame_idx / self.video_info.fps,
                        team=self.team_home
                    )
                    self.formation_history[self.team_home].append(formation_data)
                    
                    # Track formation transitions
                    if len(self.formation_history[self.team_home]) > 1:
                        prev_formation = self.formation_history[self.team_home][-2].shape
                        current_formation = formation
                        if prev_formation != current_formation:
                            transition_key = f"{prev_formation}->{current_formation}"
                            self.formation_transitions[self.team_home][transition_key] += 1
                
                self.team_formations[self.team_home][formation] += 1
                
                # Draw formation lines on frame
                self.draw_formation_lines(frame, home_players, 
                                        tuple(int(self.home_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)))
            
            # Process away team formation  
            if len(away_players) > 5:
                self.team_positions[self.team_away].append([p.center for p in away_players])
                
                # Determine formation using the selected method
                if self.formation_detection_method == "Basic":
                    formation, confidence = self.calculate_formation_basic(away_players)
                elif self.formation_detection_method == "Clustering":
                    formation, confidence = self.calculate_formation_clustering(away_players)
                else:  # Enhanced method
                    formation, confidence, positions = self.calculate_formation_enhanced(away_players)
                    
                    # Store formation history with timestamp
                    formation_data = Formation(
                        shape=formation,
                        positions=positions,
                        confidence=confidence,
                        timestamp=frame_idx / self.video_info.fps,
                        team=self.team_away
                    )
                    self.formation_history[self.team_away].append(formation_data)
                    
                    # Track formation transitions
                    if len(self.formation_history[self.team_away]) > 1:
                        prev_formation = self.formation_history[self.team_away][-2].shape
                        current_formation = formation
                        if prev_formation != current_formation:
                            transition_key = f"{prev_formation}->{current_formation}"
                            self.formation_transitions[self.team_away][transition_key] += 1
                
                self.team_formations[self.team_away][formation] += 1
                
                # Draw formation lines
                self.draw_formation_lines(frame, away_players, 
                                         tuple(int(self.away_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0)))
        except Exception as e:
            self.st.warning(f"Error in formation analysis: {str(e)}")
            traceback.print_exc()
    
    def calculate_formation_basic(self, players):
        """Basic method to calculate team formation based on player positions"""
        try:
            # Sort players by y-coordinate (vertical position)
            sorted_players = sorted(players, key=lambda p: p.center[1])
            
            # Count players in different thirds of the field (defense, midfield, attack)
            y_positions = [p.center[1] for p in sorted_players]
            y_min, y_max = min(y_positions), max(y_positions)
            range_y = y_max - y_min if y_max > y_min else 1
            
            defenders = sum(1 for p in sorted_players if (p.center[1] - y_min) / range_y < 0.33)
            midfielders = sum(1 for p in sorted_players if 0.33 <= (p.center[1] - y_min) / range_y < 0.66)
            attackers = sum(1 for p in sorted_players if (p.center[1] - y_min) / range_y >= 0.66)
            
            # Return formation as string (e.g., "4-3-3") and confidence value
            confidence = 0.7  # Basic method has a lower confidence by default
            return f"{defenders}-{midfielders}-{attackers}", confidence
        except Exception as e:
            print(f"Error in basic formation calculation: {e}")
            return "4-4-2", 0.5  # Default fallback
    
    def calculate_formation_clustering(self, players):
        """Calculate formation using clustering algorithms"""
        try:
            # Get player positions
            positions = np.array([list(p.center) for p in players])
            
            if len(positions) < 5:  # Need at least 5 players for meaningful clustering
                return "Unknown", 0.4
            
            # Normalize positions to 0-1 range for better clustering
            x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            range_x = x_max - x_min if x_max > x_min else 1
            range_y = y_max - y_min if y_max > y_min else 1
            
            normalized_positions = positions.copy()
            normalized_positions[:, 0] = (positions[:, 0] - x_min) / range_x
            normalized_positions[:, 1] = (positions[:, 1] - y_min) / range_y
            
            # Apply K-means clustering with k=3 for defense, midfield, attack
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(normalized_positions)
            
            # Count players in each cluster
            cluster_counts = np.bincount(clusters, minlength=3)
            
            # Sort clusters by y-position (vertical position)
            cluster_y_means = [np.mean(normalized_positions[clusters == i, 1]) for i in range(3)]
            sorted_indices = np.argsort(cluster_y_means)
            
            # Get player counts from back to front
            defenders = cluster_counts[sorted_indices[0]]
            midfielders = cluster_counts[sorted_indices[1]]
            attackers = cluster_counts[sorted_indices[2]]
            
            # Calculate confidence based on cluster separation
            confidence = min(1.0, 0.5 + kmeans.inertia_ / len(positions))
            
            # Return formation as string
            return f"{defenders}-{midfielders}-{attackers}", confidence
            
        except Exception as e:
            print(f"Clustering error: {str(e)}")
            # Fallback to basic method
            return self.calculate_formation_basic(players)
    
    def calculate_formation_enhanced(self, players):
        """Advanced formation detection with player role identification"""
        try:
            # Get player positions
            positions = np.array([list(p.center) for p in players])
            
            if len(positions) < 7:  # Need enough players for meaningful analysis
                return "Unknown", 0.4, positions.tolist()
            
            # Determine field orientation (which direction team is attacking)
            # For simplicity, we'll assume left-to-right is the attacking direction
            # This would need to be adjusted based on actual game context
            attacking_left_to_right = True
            
            # Normalize positions to 0-1 range for better clustering
            x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
            y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            range_x = x_max - x_min if x_max > x_min else 1
            range_y = y_max - y_min if y_max > y_min else 1
            
            normalized_positions = positions.copy()
            normalized_positions[:, 0] = (positions[:, 0] - x_min) / range_x
            normalized_positions[:, 1] = (positions[:, 1] - y_min) / range_y
            
            # Identify goalkeeper (usually the player furthest back)
            gk_idx = np.argmin(normalized_positions[:, 0]) if attacking_left_to_right else np.argmax(normalized_positions[:, 0])
            
            # Remove goalkeeper from formation analysis
            outfield_positions = np.delete(normalized_positions, gk_idx, axis=0)
            original_positions = np.delete(positions, gk_idx, axis=0)
            
            # Apply DBSCAN clustering to potentially identify lines
            db = DBSCAN(eps=0.15, min_samples=2)
            clusters = db.fit_predict(outfield_positions)
            
            # If DBSCAN fails to find good clusters, fall back to K-means
            if len(np.unique(clusters[clusters >= 0])) < 2:
                # Try K-means with variable number of clusters
                best_inertia = float('inf')
                best_k = 3
                best_labels = None
                
                for k in range(3, 6):  # Try 3, 4, and 5 clusters
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(outfield_positions)
                    
                    if kmeans.inertia_ < best_inertia:
                        best_inertia = kmeans.inertia_
                        best_k = k
                        best_labels = kmeans.labels_
                
                clusters = best_labels
            
            # Count players in each cluster or line
            unique_clusters = np.unique(clusters)
            cluster_counts = []
            
            for c in unique_clusters:
                if c >= 0:  # Ignore noise points labeled as -1
                    count = np.sum(clusters == c)
                    cluster_counts.append((c, count))
            
            # Sort clusters by average x-position (for attacking direction)
            cluster_x_means = []
            for c, _ in cluster_counts:
                mean_x = np.mean(outfield_positions[clusters == c, 0])
                cluster_x_means.append((c, mean_x))
            
            # Sort from defense to attack
            if attacking_left_to_right:
                cluster_x_means.sort(key=lambda x: x[1])
            else:
                cluster_x_means.sort(key=lambda x: -x[1])
            
            # Organize players into lines
            lines = []
            for c, _ in cluster_x_means:
                line_size = np.sum(clusters == c)
                lines.append(int(line_size))
            
            # Append goalkeeper to the formation
            lines = [1] + lines  # Add goalkeeper as "1"
            
            # Limit to common football formations with sanity checks
            if len(lines) < 3:
                # Not enough lines detected, fallback to simpler method
                formation, conf = self.calculate_formation_clustering(players)
                return formation, conf, positions.tolist()
            
            # Convert to standard formation string (ignoring goalkeeper)
            formation_str = "-".join(str(l) for l in lines[1:])
            
            # Calculate confidence based on clustering quality
            if len(np.unique(clusters)) <= 1:
                confidence = 0.5  # Low confidence if clustering failed
            else:
                # Higher confidence with more distinct lines
                confidence = min(0.9, 0.6 + 0.1 * len(np.unique(clusters)))
            
            # Map to common formations if close
            common_formations = {
                "4-4-2": 0,
                "4-3-3": 0,
                "3-5-2": 0,
                "5-3-2": 0,
                "4-2-3-1": 0,
                "3-4-3": 0
            }
            
            for common in common_formations:
                common_parts = common.split("-")
                if len(common_parts) == len(lines) - 1:
                    similarity = sum(abs(int(common_parts[i]) - lines[i+1]) for i in range(len(common_parts)))
                    if similarity <= 2:  # Allow small variations
                        formation_str = common
                        confidence = max(confidence, 0.8)
                        break
            
            return formation_str, confidence, positions.tolist()
            
        except Exception as e:
            print(f"Enhanced formation detection error: {str(e)}")
            traceback.print_exc()
            # Fallback to clustering method
            formation, conf = self.calculate_formation_clustering(players)
            return formation, conf, positions.tolist()
    
    def draw_formation_lines(self, frame, players, color):
        """Draw lines connecting players in the same line (defense, midfield, attack)"""
        try:
            # Sort players by y-coordinate (vertical position)
            sorted_players = sorted(players, key=lambda p: p.center[1])
            
            # Group players into lines
            y_positions = [p.center[1] for p in sorted_players]
            y_min, y_max = min(y_positions), max(y_positions)
            range_y = y_max - y_min if y_max > y_min else 1
            
            defenders = [p for p in sorted_players if (p.center[1] - y_min) / range_y < 0.33]
            midfielders = [p for p in sorted_players if 0.33 <= (p.center[1] - y_min) / range_y < 0.66]
            attackers = [p for p in sorted_players if (p.center[1] - y_min) / range_y >= 0.66]
            
            # Sort each line by x-coordinate
            for line in [defenders, midfielders, attackers]:
                if len(line) > 1:
                    line.sort(key=lambda p: p.center[0])
                    
                    # Draw lines connecting players in the same line
                    for i in range(len(line) - 1):
                        pt1 = (int(line[i].center[0]), int(line[i].center[1]))
                        pt2 = (int(line[i+1].center[0]), int(line[i+1].center[1]))
                        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
                    
                    # Draw line name and role
                    if line:
                        line_type = "DEF" if line == defenders else "MID" if line == midfielders else "ATT"
                        avg_x = sum(p.center[0] for p in line) / len(line)
                        avg_y = sum(p.center[1] for p in line) / len(line)
                        cv2.putText(frame, f"{line_type} ({len(line)})", 
                                   (int(avg_x), int(avg_y - 15)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"Error drawing formation lines: {str(e)}")
    
    def detect_events(self, frame_idx, players, ball_position):
        """Detect key events like passes and shots with enhanced accuracy"""
        try:
            # Need at least a few frames of ball positions
            if len(self.ball_positions) < 5:
                return
            
            # Calculate ball velocity vector
            prev_ball_positions = list(self.ball_positions)[-5:]
            ball_vector = (
                ball_position[0] - prev_ball_positions[0][0],
                ball_position[1] - prev_ball_positions[0][1]
            )
            ball_speed = np.sqrt(ball_vector[0]**2 + ball_vector[1]**2)
            
            # Get nearest players to current and previous ball positions
            if players:
                current_nearest = min(players, key=lambda p: 
                                    distance.euclidean(p.center, ball_position))
                prev_nearest = min(players, key=lambda p: 
                                  distance.euclidean(p.center, prev_ball_positions[0]))
                
                # Enhanced pass detection with sensitivity setting
                pass_speed_threshold = 5 * (self.pass_detection_sensitivity / 5)  # Adjust based on sensitivity
                player_distance_threshold = 5 * (10 / self.pass_detection_sensitivity)  # Inverse relationship
                
                # Detect pass - ball moved between players of same team
                if (current_nearest.track_id != prev_nearest.track_id and 
                    current_nearest.team == prev_nearest.team and 
                    ball_speed > pass_speed_threshold and
                    distance.euclidean(current_nearest.center, prev_nearest.center) > player_distance_threshold):
                    
                    # Calculate pass length in meters
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    pass_length = distance.euclidean(ball_position, prev_ball_positions[0]) * pixel_to_meter
                    
                    # Determine pass zones
                    zone_from_x = min(int(prev_ball_positions[0][0] / self.video_info.target_width * 9), 8)
                    zone_from_y = min(int(prev_ball_positions[0][1] / self.video_info.target_height * 6), 5)
                    
                    zone_to_x = min(int(ball_position[0] / self.video_info.target_width * 9), 8)
                    zone_to_y = min(int(ball_position[1] / self.video_info.target_height * 6), 5)
                    
                    # Determine pass direction (forward, backward, lateral)
                    dx = ball_position[0] - prev_ball_positions[0][0]
                    dy = ball_position[1] - prev_ball_positions[0][1]
                    
                    # Simplify direction based on dominant axis
                    if abs(dx) > abs(dy):
                        direction = "forward" if dx > 0 else "backward"
                    else:
                        direction = "lateral"
                    
                    # Determine pass type based on trajectory and speed
                    if ball_speed > 20:
                        pass_type = "through" if direction == "forward" else "long"
                    elif abs(ball_vector[1]) > abs(ball_vector[0]) * 2:
                        pass_type = "lofted"
                    else:
                        pass_type = "ground"
                    
                    # Determine if pass is progressive (moves ball significantly forward)
                    progressive = direction == "forward" and dx > self.video_info.target_width / 5
                    
                    # Determine if pass is into danger zone (final third or penalty area)
                    danger_zone = False
                    field_third = self.video_info.target_width / 3
                    
                    # If passing into final third or penalty area
                    if (zone_to_x >= 6 or  # Final third of pitch
                        (zone_to_x >= 7 and zone_to_y >= 2 and zone_to_y <= 3)):  # Penalty area
                        danger_zone = True
                    
                    # Determine if pass breaks defensive lines (simplified)
                    # Find players from opposite team between passer and receiver
                    opp_team = self.team_away if current_nearest.team == self.team_home else self.team_home
                    opp_players = [p for p in players if p.team == opp_team]
                    
                    # Check if pass goes between/through opposing players
                    breaking_lines = False
                    if opp_players:
                        # Create a simplified line from passer to receiver
                        pass_dir_x = ball_position[0] - prev_ball_positions[0][0]
                        pass_dir_y = ball_position[1] - prev_ball_positions[0][1]
                        pass_length_pixels = np.sqrt(pass_dir_x**2 + pass_dir_y**2)
                        
                        # Check if any opposing players are near the pass line but not too close to passer/receiver
                        for opp in opp_players:
                            # Distance from opponent to pass line
                            d = self.point_to_line_distance(
                                opp.center, 
                                prev_ball_positions[0], 
                                ball_position
                            )
                            
                            # Check if opponent is near the pass line
                            if d < 30:  # Pixels threshold
                                breaking_lines = True
                                break
                    
                    # Determine if pass is a switch of play (cross-field pass)
                    switch_play = abs(dy) > self.video_info.target_height / 3
                    
                    # Calculate expected assists (xA) - simplified model
                    # Higher xA for passes into dangerous areas or that create shooting opportunities
                    xA = 0.0
                    if danger_zone:
                        if pass_type == "through":
                            xA = 0.15  # Through passes into danger zone have higher xA
                        else:
                            xA = 0.05  # Other passes into danger zone
                    elif progressive:
                        xA = 0.02  # Progressive passes have some xA value
                    
                    # Track pass length distribution for analysis
                    # Categorize passes as short, medium, or long
                    if pass_length < 10:
                        length_category = "short"
                    elif pass_length < 25:
                        length_category = "medium"
                    else:
                        length_category = "long"
                    
                    self.pass_length_distribution[current_nearest.team][length_category] += 1
                    
                    # Create pass event
                    pass_event = PassEvent(
                        time=frame_idx / self.video_info.fps,
                        from_player=prev_nearest.track_id,
                        to_player=current_nearest.track_id,
                        team=current_nearest.team,
                        from_position=prev_ball_positions[0],
                        to_position=ball_position,
                        completed=True,
                        length=pass_length,
                        zone_from=(zone_from_x, zone_from_y),
                        zone_to=(zone_to_x, zone_to_y),
                        direction=direction,
                        pass_type=pass_type,
                        progressive=progressive,
                        danger_zone=danger_zone,
                        breaking_lines=breaking_lines,
                        switch_play=switch_play,
                        xA=xA
                    )
                    
                    # Add to pass data
                    self.pass_data.append(asdict(pass_event))
                    
                    # Update pass statistics
                    self.pass_types[current_nearest.team][pass_type] += 1
                    self.pass_directions[current_nearest.team][direction] += 1
                    
                    # Track progressive passes and danger zone passes
                    if progressive:
                        self.progressive_passes[current_nearest.team] += 1
                    if danger_zone:
                        self.danger_zone_passes[current_nearest.team] += 1
                    if breaking_lines:
                        self.breaking_lines_passes[current_nearest.team] += 1
                    if switch_play:
                        self.switch_play_passes[current_nearest.team] += 1
                    
                    # Add xA to total
                    self.total_xA[current_nearest.team] += xA
                    
                    # Update pass network
                    self.pass_networks[current_nearest.team][prev_nearest.track_id][current_nearest.track_id] += 1
                    
                    # Update zone statistics
                    team_idx = 0 if current_nearest.team == self.team_home else 1
                    self.zone_passes[zone_from_y, zone_from_x, team_idx] += 1
                    
                    # Add to general events list
                    self.events.append({
                        'time': frame_idx / self.video_info.fps,
                        'type': 'pass',
                        'from_player': prev_nearest.track_id,
                        'to_player': current_nearest.track_id,
                        'team': current_nearest.team,
                        'pass_type': pass_type,
                        'direction': direction,
                        'progressive': progressive,
                        'danger_zone': danger_zone,
                        'breaking_lines': breaking_lines,
                        'switch_play': switch_play,
                        'length': pass_length,
                        'xA': xA
                    })
                
                # Enhanced shot detection with improved sensitivity
                goal_line_left = 0
                goal_line_right = self.video_info.target_width
                goal_center_left = (goal_line_left, self.video_info.target_height / 2)
                goal_center_right = (goal_line_right, self.video_info.target_height / 2)
                
                # Calculate shot threshold based on sensitivity
                shot_speed_threshold = 15 * (self.shot_detection_sensitivity / 5)
                shot_angle_threshold = 30 * (10 / self.shot_detection_sensitivity)
                
                # Check if ball is moving toward either goal with high speed
                angle_to_left = self.angle_between_vectors(
                    ball_vector, 
                    (goal_center_left[0] - ball_position[0], goal_center_left[1] - ball_position[1])
                )
                
                angle_to_right = self.angle_between_vectors(
                    ball_vector, 
                    (goal_center_right[0] - ball_position[0], goal_center_right[1] - ball_position[1])
                )
                
                is_shot_left = ball_speed > shot_speed_threshold and angle_to_left < shot_angle_threshold
                is_shot_right = ball_speed > shot_speed_threshold and angle_to_right < shot_angle_threshold
                
                if is_shot_left or is_shot_right:
                    target_goal = self.team_away if is_shot_left else self.team_home
                    
                    # Calculate shot distance from goal
                    goal_position = goal_center_left if is_shot_left else goal_center_right
                    pixel_to_meter = (FIELD_WIDTH / self.video_info.target_width + FIELD_HEIGHT / self.video_info.target_height) / 2
                    shot_distance = distance.euclidean(prev_ball_positions[0], goal_position) * pixel_to_meter
                    
                    # Calculate shot angle from goal center
                    shot_angle = self.angle_between_vectors(
                        (goal_position[0] - prev_ball_positions[0][0], goal_position[1] - prev_ball_positions[0][1]),
                        (1, 0) if is_shot_left else (-1, 0)  # Perpendicular to goal line
                    )
                    
                    # Determine if shot is on target (simplified)
                    on_target = ball_speed > 25 and shot_angle < 15
                    
                    # Calculate expected goal (xG) value based on distance and angle
                    # Enhanced xG model with distance and angle factors
                    distance_factor = 1 / (1 + (shot_distance/10)**2)  # Decay with square of distance
                    angle_factor = 1 - (min(shot_angle, 90) / 90)**2  # Penalty for wide angles
                    
                    # Baseline xG
                    xg_baseline = 0.3 * distance_factor * angle_factor
                    
                    # Adjust for location
                    penalty_box = shot_distance < 18 and shot_angle < 45
                    six_yard_box = shot_distance < 6 and shot_angle < 30
                    
                    if six_yard_box:
                        xg_location_factor = 2.5  # Much higher xG in six-yard box
                    elif penalty_box:
                        xg_location_factor = 1.5  # Higher xG in penalty box
                    else:
                        xg_location_factor = 1.0
                    
                    # Final xG calculation with bounds
                    xg = min(0.9, max(0.01, xg_baseline * xg_location_factor))
                    
                    # Determine shot type based on ball position and speed
                    shot_type = "normal"
                    if ball_position[1] < (prev_ball_positions[0][1] - 20):  # Ball moved upward significantly
                        shot_type = "volley"
                    
                    # Get other nearby players to determine pressure
                    other_players = [p for p in players if p.track_id != prev_nearest.track_id]
                    if other_players:
                        min_distance = min(distance.euclidean(p.center, prev_ball_positions[0]) for p in other_players)
                        # Normalize pressure from 0-1 based on distance
                        pressure = max(0, min(1.0, 30.0 / max(min_distance, 1.0)))
                    else:
                        pressure = 0
                    
                    # Determine shot scenario
                    # For simplicity, we'll use simple heuristics
                    scenario = "open_play"
                    
                    # Determine shot zone
                    shot_position = prev_ball_positions[0]
                    center_y = self.video_info.target_height / 2
                    if abs(shot_position[1] - center_y) < self.video_info.target_height / 6:
                        zone = "central"
                    elif shot_position[1] < center_y:
                        zone = "left_side"
                    else:
                        zone = "right_side"
                    
                    # Further refine with distance
                    if shot_distance < 18:  # Penalty box distance in meters
                        zone = f"{zone}_box"
                    else:
                        zone = f"{zone}_outside_box"
                    
                    # Create shot event
                    shot_event = ShotEvent(
                        time=frame_idx / self.video_info.fps,
                        player=prev_nearest.track_id,
                        team=prev_nearest.team,
                        position=prev_ball_positions[0],
                        target_goal=target_goal,
                        on_target=on_target,
                        goal=False,  # Would need additional detection for goals
                        expected_goal=xg,
                        distance=shot_distance,
                        angle=shot_angle,
                        shot_type=shot_type,
                        scenario=scenario,
                        pressure=pressure,
                        zone=zone
                    )
                    
                    # Add to shot data
                    self.shot_data.append(asdict(shot_event))
                    
                    # Update shot statistics
                    team = prev_nearest.team
                    self.shot_types[team][shot_type] += 1
                    self.shot_scenarios[team][scenario] += 1
                    self.shot_zones[team][zone] += 1
                    self.total_xG[team] += xg
                    
                    if pressure > 0.5:  # Threshold for "under pressure"
                        self.shots_under_pressure[team] += 1
                    
                    # Update zone statistics
                    zone_x = min(int(prev_ball_positions[0][0] / self.video_info.target_width * 9), 8)
                    zone_y = min(int(prev_ball_positions[0][1] / self.video_info.target_height * 6), 5)
                    team_idx = 0 if team == self.team_home else 1
                    self.zone_shots[zone_y, zone_x, team_idx] += 1
                    
                    # Add to general events list
                    self.events.append({
                        'time': frame_idx / self.video_info.fps,
                        'type': 'shot',
                        'player': prev_nearest.track_id,
                        'team': prev_nearest.team,
                        'target_goal': target_goal,
                        'on_target': on_target,
                        'xG': xg,
                        'shot_type': shot_type,
                        'scenario': scenario,
                        'pressure': pressure,
                        'zone': zone,
                        'distance': shot_distance,
                        'angle': shot_angle
                    })
        except Exception as e:
            print(f"Error in event detection: {str(e)}")
            traceback.print_exc()

    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the shortest distance from a point to a line segment"""
        x1, y1 = line_start
        x2, y2 = line_end
        x0, y0 = point
        
        # Length of line segment squared
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0:  # Line segment is a point
            return np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        # Consider the line extending the segment, parameterized as start + t (end - start)
        # Find projection of point onto the line
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / l2))
        
        # Calculate the projection point
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        # Distance from point to projection
        return np.sqrt((x0 - proj_x)**2 + (y0 - proj_y)**2)
    
    def angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        try:
            v1_norm = np.sqrt(v1[0]**2 + v1[1]**2)
            v2_norm = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if v1_norm == 0 or v2_norm == 0:
                return 0
                
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            cos_angle = dot_product / (v1_norm * v2_norm)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle_rad)
        except Exception as e:
            print(f"Error calculating angle: {str(e)}")
            return 0
    
    def prepare_analysis_results(self):
        """Prepare analysis results for display"""
        try:
            # Calculate player statistics
            player_stats = {}
            for player_id in self.player_positions.keys():
                if player_id not in self.player_team:
                    continue
                    
                team = self.player_team[player_id]
                
                # Calculate average speed
                avg_speed = np.mean(self.speed_data[player_id]) if self.speed_data[player_id] else 0
                max_speed = np.max(self.speed_data[player_id]) if self.speed_data[player_id] else 0
                
                # Calculate possession percentage
                total_frames = max(sum(self.team_possession_frames.values()), 1)  # Avoid division by zero
                possession_percentage = (self.ball_possession[player_id] / total_frames * 100) if total_frames > 0 else 0
                
                # Enhanced pass statistics
                passes_completed = sum(1 for p in self.pass_data if p['from_player'] == player_id)
                passes_received = sum(1 for p in self.pass_data if p['to_player'] == player_id)
                
                # Pass completion rate
                pass_completion_rate = 100  # Default to 100% if no data
                
                # Calculate player influence (based on pass network centrality - simplified)
                player_influence = passes_completed + passes_received
                
                # Calculate progressive passes
                progressive_passes = sum(1 for p in self.pass_data if p['from_player'] == player_id and p.get('progressive', False))
                
                # Calculate breaking lines passes
                breaking_lines_passes = sum(1 for p in self.pass_data if p['from_player'] == player_id and p.get('breaking_lines', False))
                
                # Calculate expected assists (xA)
                player_xA = sum(p.get('xA', 0) for p in self.pass_data if p['from_player'] == player_id)
                
                # Enhanced shot statistics
                shots = sum(1 for s in self.shot_data if s['player'] == player_id)
                shots_on_target = sum(1 for s in self.shot_data if s['player'] == player_id and s.get('on_target', False))
                
                # Expected goals (xG)
                player_xG = sum(s.get('expected_goal', 0) for s in self.shot_data if s['player'] == player_id)
                
                # Store player stats
                player_stats[player_id] = {
                    'Player ID': player_id,
                    'Team': team,
                    'Distance (m)': round(self.distance_data[player_id], 2),
                    'Avg Speed (m/s)': round(avg_speed, 2),
                    'Max Speed (m/s)': round(max_speed, 2),
                    'Possession (%)': round(possession_percentage, 2),
                    'Passes': passes_completed,
                    'Passes Received': passes_received,
                    'Pass Completion (%)': round(pass_completion_rate, 2),
                    'Progressive Passes': progressive_passes,
                    'Breaking Lines Passes': breaking_lines_passes,
                    'Expected Assists (xA)': round(player_xA, 3),
                    'Shots': shots,
                    'Shots on Target': shots_on_target,
                    'Expected Goals (xG)': round(player_xG, 3),
                    'Influence': player_influence
                }
            
            # Store in session state
            self.st.session_state[KEY_PLAYER_STATS] = player_stats
            
            # Create DataFrame for visualization
            self.player_stats_df = pd.DataFrame.from_dict(player_stats, orient='index') if player_stats else pd.DataFrame()
            
            # Calculate team statistics
            total_frames = max(sum(self.team_possession_frames.values()), 1)  # Avoid division by zero
            
            # Calculate pass success rate
            for team in [self.team_home, self.team_away]:
                total_passes = sum(1 for p in self.pass_data if p['team'] == team)
                completed_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('completed', True))
                self.pass_success_rate[team] = (completed_passes / total_passes * 100) if total_passes > 0 else 0
            
            # Calculate shot success rate
            for team in [self.team_home, self.team_away]:
                total_shots = sum(1 for s in self.shot_data if s['team'] == team)
                shots_on_target = sum(1 for s in self.shot_data if s['team'] == team and s.get('on_target', True))
                self.shot_success_rate[team] = (shots_on_target / total_shots * 100) if total_shots > 0 else 0
            
            # Enhanced team stats
            self.team_stats = {
                self.team_home: {
                    'Possession (%)': round(self.team_possession_frames[self.team_home] / total_frames * 100, 2) if total_frames > 0 else 0,
                    'Distance (m)': round(sum(self.distance_data[p] for p, team in self.player_team.items() if team == self.team_home), 2),
                    'Passes': sum(1 for p in self.pass_data if p['team'] == self.team_home),
                    'Pass Completion (%)': round(self.pass_success_rate[self.team_home], 2),
                    'Forward Passes (%)': round(sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'forward') / 
                                 max(sum(1 for p in self.pass_data if p['team'] == self.team_home), 1) * 100, 2),
                    'Progressive Passes': self.progressive_passes[self.team_home],
                    'Breaking Lines Passes': self.breaking_lines_passes[self.team_home],
                    'Danger Zone Passes': self.danger_zone_passes[self.team_home],
                    'Expected Assists (xA)': round(self.total_xA[self.team_home], 3),
                    'Shots': sum(1 for s in self.shot_data if s['team'] == self.team_home),
                    'Shots on Target': sum(1 for s in self.shot_data if s['team'] == self.team_home and s.get('on_target', False)),
                    'Shot Accuracy (%)': round(self.shot_success_rate[self.team_home], 2),
                    'Shots Under Pressure': self.shots_under_pressure[self.team_home],
                    'Expected Goals (xG)': round(self.total_xG[self.team_home], 3),
                    'Most Used Formation': max(self.team_formations[self.team_home].items(), key=lambda x: x[1])[0] if self.team_formations[self.team_home] else "N/A"
                },
                self.team_away: {
                    'Possession (%)': round(self.team_possession_frames[self.team_away] / total_frames * 100, 2) if total_frames > 0 else 0,
                    'Distance (m)': round(sum(self.distance_data[p] for p, team in self.player_team.items() if team == self.team_away), 2),
                    'Passes': sum(1 for p in self.pass_data if p['team'] == self.team_away),
                    'Pass Completion (%)': round(self.pass_success_rate[self.team_away], 2),
                    'Forward Passes (%)': round(sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'forward') / 
                                 max(sum(1 for p in self.pass_data if p['team'] == self.team_away), 1) * 100, 2),
                    'Progressive Passes': self.progressive_passes[self.team_away],
                    'Breaking Lines Passes': self.breaking_lines_passes[self.team_away],
                    'Danger Zone Passes': self.danger_zone_passes[self.team_away],
                    'Expected Assists (xA)': round(self.total_xA[self.team_away], 3),
                    'Shots': sum(1 for s in self.shot_data if s['team'] == self.team_away),
                    'Shots on Target': sum(1 for s in self.shot_data if s['team'] == self.team_away and s.get('on_target', False)),
                    'Shot Accuracy (%)': round(self.shot_success_rate[self.team_away], 2),
                    'Shots Under Pressure': self.shots_under_pressure[self.team_away],
                    'Expected Goals (xG)': round(self.total_xG[self.team_away], 3),
                    'Most Used Formation': max(self.team_formations[self.team_away].items(), key=lambda x: x[1])[0] if self.team_formations[self.team_away] else "N/A"
                }
            }
            
            # Store team stats in session state
            self.st.session_state[KEY_TEAM_STATS] = self.team_stats
            
            # Create zones data for heatmap
            total_zone_time = np.sum(self.zone_possession)
            if total_zone_time > 0:
                self.zone_percentage = self.zone_possession / total_zone_time * 100
            else:
                self.zone_percentage = np.zeros_like(self.zone_possession)
                
            # Store events in session state
            self.st.session_state[KEY_EVENTS] = self.events
        except Exception as e:
            self.st.error(f"Error preparing analysis results: {str(e)}")
            traceback.print_exc()
    def generate_llm_match_report(self):
        """Generate detailed match report using the already-configured Gemini instance"""
        try:
            prompt = f"""
            You are an elite football analyst. Write a formal, detailed match report between {self.team_home} and {self.team_away}.

            Team Stats:
            {json.dumps(self.team_stats, indent=2)}

            Top Player Stats:
            {self.player_stats_df.head(5).to_dict(orient='records')}

            Include:
            - Tactical analysis
            - Performance highlights
            - Key players
            - Areas of improvement
            - Suggestions for both teams
            """

            response = self.gemini_model_instance.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            return f"(⚠️ LLM report generation failed: {str(e)})"

            
    def analyze_strengths_weaknesses(self):
        """Analyze team strengths and weaknesses based on collected data"""
        try:
            for team in [self.team_home, self.team_away]:
                other_team = self.team_away if team == self.team_home else self.team_home
                
                # Get team metrics
                possession_pct = self.team_stats[team]['Possession (%)']
                total_passes = max(sum(1 for p in self.pass_data if p['team'] == team), 1)  # Avoid division by zero
                completed_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('completed', True))
                pass_completion = (completed_passes / total_passes * 100) if total_passes > 0 else 0
                
                # Get pass metrics
                pass_completion = self.pass_success_rate[team]
                forward_pass_pct = self.team_stats[team]['Forward Passes (%)']
                progressive_passes = self.progressive_passes[team]
                danger_zone_passes = self.danger_zone_passes[team]
                breaking_lines_passes = self.breaking_lines_passes[team]
                xA = self.total_xA[team]
                
                # Get shot metrics
                total_shots = sum(1 for s in self.shot_data if s['team'] == team)
                shots_on_target = sum(1 for s in self.shot_data if s['team'] == team and s.get('on_target', False))
                shot_accuracy = (shots_on_target / total_shots * 100) if total_shots > 0 else 0
                shots_under_pressure = self.shots_under_pressure[team]
                total_xG = self.total_xG[team]
                
                # Get defensive metrics
                defensive_actions_count = sum(len(actions) for player_id, actions in self.defensive_actions.items() 
                                          if player_id in self.player_team and self.player_team[player_id] == team)
                
                # Team strengths
                strengths = []
                
                # Possession strengths
                if possession_pct > 55:
                    strengths.append(("high_possession_percentage", possession_pct))
                
                # Passing strengths
                if pass_completion > 80:
                    strengths.append(("high_pass_completion", pass_completion))
                
                if forward_pass_pct > 40:
                    strengths.append(("high_forward_passes", forward_pass_pct))
                
                if progressive_passes > 20:
                    strengths.append(("high_progressive_passes", progressive_passes))
                
                if breaking_lines_passes > 15:
                    strengths.append(("high_through_balls", breaking_lines_passes))
                
                if danger_zone_passes > 10:
                    strengths.append(("high_passes_final_third", danger_zone_passes))
                
                if xA > 0.8:
                    strengths.append(("high_xA", xA*100))  # Scale up for visual display
                
                # Shooting strengths
                if shot_accuracy > 40:
                    strengths.append(("high_shots_on_target", shot_accuracy))
                
                if total_xG > 1.0:
                    strengths.append(("high_xG_per_shot", total_xG*100))  # Scale up for visual display
                
                # Defensive strengths 
                if defensive_actions_count > 50:
                    strengths.append(("high_defensive_duels_won", defensive_actions_count))
                
                # Add pressing intensity if available
                pressing_success = max(self.pressing_intensity.get(team, 0), 1)
                if pressing_success > 70:
                    strengths.append(("high_pressing_success", pressing_success))
                
                # If no strengths detected, add some default ones based on best metrics
                if not strengths:
                    # Add possession as a strength (even if it's not high)
                    strengths.append(("high_possession_percentage", max(possession_pct, 1)))
                    # Add pass completion
                    strengths.append(("high_pass_completion", max(pass_completion, 1)))
                    # Add defensive actions
                    strengths.append(("high_defensive_duels_won", max(defensive_actions_count, 1)))
                
                # Sort strengths by value and take top 3
                strengths.sort(key=lambda x: x[1], reverse=True)
                self.team_strengths[team] = {
                    key: {
                        "value": value,
                        "description": self.tactical_strengths.get(key, "Team strength")
                    } for key, value in strengths[:3]
                }
                
                # Team weaknesses
                weaknesses = []
                
                # Possession weaknesses
                if possession_pct < 45:
                    weaknesses.append(("low_possession_percentage", max(possession_pct, 1)))
                
                # Passing weaknesses
                if pass_completion < 70:
                    weaknesses.append(("low_pass_completion", max(pass_completion, 1)))
                
                if forward_pass_pct < 30:
                    weaknesses.append(("low_forward_passes", max(forward_pass_pct, 1)))
                
                if progressive_passes < 10:
                    weaknesses.append(("low_progressive_passes", max(progressive_passes, 1)))
                
                # Shooting weaknesses
                if shot_accuracy < 30:
                    weaknesses.append(("low_shots_on_target", max(shot_accuracy, 1)))
                
                if total_xG < 0.5:
                    weaknesses.append(("low_xG_per_shot", max(total_xG*100, 1)))  # Scale up for display
                
                # Defensive weaknesses
                if defensive_actions_count < 30:
                    weaknesses.append(("low_defensive_duels_won", max(defensive_actions_count, 1)))
                
                if pressing_success < 50:
                    weaknesses.append(("low_pressing_success", max(pressing_success, 1)))
                
                # Calculate lateral passes
                lateral_passes = sum(1 for p in self.pass_data if p['team'] == team and p.get('direction') == 'lateral')
                lateral_pass_pct = (lateral_passes / total_passes * 100) if total_passes > 0 else 0
                
                if lateral_pass_pct > 40:
                    weaknesses.append(("high_lateral_passes", lateral_pass_pct))
                
                # Check if team is vulnerable during defensive transitions
                if shots_under_pressure > 5:
                    weaknesses.append(("poor_defensive_transitions", shots_under_pressure))
                
                # If no weaknesses detected, add some default ones
                if not weaknesses:
                    # Choose metrics that are relatively lower compared to others
                    if possession_pct < pass_completion:
                        weaknesses.append(("low_possession_percentage", max(possession_pct, 1)))
                    else:
                        weaknesses.append(("low_pass_completion", max(pass_completion, 1)))
                    
                    if shot_accuracy < 50:
                        weaknesses.append(("low_shots_on_target", max(shot_accuracy, 1)))
                    
                    weaknesses.append(("low_pressing_success", max(pressing_success, 1)))
                
                # Sort weaknesses by value (lower is worse)
                weaknesses.sort(key=lambda x: x[1])
                self.team_weaknesses[team] = {
                    key: {
                        "value": value,
                        "description": self.tactical_weaknesses.get(key, "Area for improvement")
                    } for key, value in weaknesses[:3]
                }
                
                # Generate tactical suggestions based on opponent's weaknesses
                self.generate_tactical_suggestions(team, other_team)
        except Exception as e:
            self.st.error(f"Error analyzing strengths and weaknesses: {str(e)}")
            traceback.print_exc()
            
    def generate_tactical_suggestions(self, team, opponent):
        """Generate tactical suggestions based on opponent weaknesses and playstyle"""
        try:
            suggestions = []
            
            # Get opponent's playstyle
            opponent_style = self.away_playstyle if opponent == self.team_away else self.home_playstyle
            
            # If using Gemini and insights are available, prioritize those
            if self.enable_gemini and self.gemini_insights[team]:
                # Filter for only suggestion content
                suggestions = [insight for insight in self.gemini_insights[team] 
                              if "suggestion" in insight.lower() or "recommendation" in insight.lower()]
                
                # If we found suggestions, use them
                if suggestions:
                    # Clean up the suggestions
                    cleaned_suggestions = []
                    for suggestion in suggestions:
                        # Split by lines and remove empty ones
                        lines = [line.strip() for line in suggestion.split('\n') if line.strip()]
                        cleaned_suggestions.extend(lines)
                    
                    # Keep only items that look like suggestions (not headers)
                    real_suggestions = [s for s in cleaned_suggestions 
                                       if not s.startswith('#') and not s.endswith(':') 
                                       and len(s) > 15]
                    
                    # If we have enough suggestions from Gemini, use them
                    if len(real_suggestions) >= 3:
                        self.tactical_suggestions[team] = real_suggestions[:5]
                        return
            
            # If we don't have Gemini suggestions or not enough, fall back to our built-in logic
            
            # Add counter-strategy suggestions based on opponent's playstyle
            if opponent_style in self.counter_strategies:
                suggestions.extend(self.counter_strategies[opponent_style])
            else:
                # Default suggestions if playstyle not recognized
                suggestions.extend(self.counter_strategies["Custom"])
            
            # Add specific suggestions based on opponent's weaknesses
            for weakness_key in self.team_weaknesses.get(opponent, {}):
                if weakness_key == "low_possession_percentage":
                    suggestions.append("Apply high pressing to force turnovers")
                elif weakness_key == "low_pass_completion":
                    suggestions.append("Press aggressively during build-up phase")
                elif weakness_key == "low_shots_on_target":
                    suggestions.append("Allow low-quality shots from distance while protecting the box")
                elif weakness_key == "low_pressing_success":
                    suggestions.append("Use technical midfielders to bypass their press")
                elif weakness_key == "low_defensive_duels_won":
                    suggestions.append("Target 1v1 situations in attack")
                elif weakness_key == "low_forward_passes":
                    suggestions.append("Block forward passing lanes to force backward/sideways passes")
                elif weakness_key == "high_lateral_passes":
                    suggestions.append("Set pressing traps on sidelines to win ball during sideways passes")
                elif weakness_key == "low_progressive_passes":
                    suggestions.append("Press aggressively in midfield to prevent ball progression")
                elif weakness_key == "poor_defensive_transitions":
                    suggestions.append("Counter-attack quickly after winning possession")
                elif weakness_key == "low_xG_per_shot":
                    suggestions.append("Compact defense to force low-quality shots from distance")
                
                # Add suggestions based on team's strengths
                for strength_key in self.team_strengths.get(team, {}):
                    if strength_key == "high_possession_percentage":
                        suggestions.append("Focus on patient build-up play to capitalize on possession advantage")
                    elif strength_key == "high_pass_completion":
                        suggestions.append("Use quick passing combinations to break through defensive lines")
                    elif strength_key == "high_shots_on_target":
                        suggestions.append("Create more shooting opportunities for forwards")
                    elif strength_key == "high_pressing_success":
                        suggestions.append("Implement aggressive pressing triggers in opponent's half")
                    elif strength_key == "high_defensive_duels_won":
                        suggestions.append("Encourage defenders to step up for interceptions")
                    elif strength_key == "high_forward_passes":
                        suggestions.append("Maintain vertical passing options to exploit progressive passing")
                    elif strength_key == "high_through_balls":
                        suggestions.append("Position forwards to make runs for through passes behind defense")
                    elif strength_key == "high_progressive_passes":
                        suggestions.append("Create space in midfield to continue progressive passing patterns")
                    elif strength_key == "high_passes_final_third":
                        suggestions.append("Position additional players in the final third to capitalize on chances")
                    elif strength_key == "high_xA" or strength_key == "high_xG_per_shot":
                        suggestions.append("Prioritize getting the ball to creative players in dangerous areas")
                
                # Ensure we have at least 5 suggestions
                if len(suggestions) < 5:
                    default_suggestions = [
                        "Focus on maintaining defensive organization",
                        "Exploit spaces in wide areas",
                        "Increase tempo in transition phases",
                        "Use defensive midfielder to screen opponent attacks",
                        "Maintain compact shape between defensive lines"
                    ]
                    # Add default suggestions until we have at least 5
                    for suggestion in default_suggestions:
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)
                        if len(suggestions) >= 5:
                            break
                
                # Store unique suggestions (up to 5)
                self.tactical_suggestions[team] = list(set(suggestions))[:5]
        except Exception as e:
            self.st.error(f"Error generating tactical suggestions: {str(e)}")
            traceback.print_exc()
            # Set default suggestions in case of error
            self.tactical_suggestions[team] = [
                "Maintain defensive organization",
                "Focus on possession in the middle third",
                "Press when the ball is in wide areas",
                "Create numerical advantages in attack",
                "Stay compact between the lines"
            ]
    
    def generate_gemini_insights(self):
        """Generate tactical insights using Gemini API"""
        if not self.st.session_state.gemini_api_key:
            self.st.warning("Gemini API key not provided. Skipping AI insights.")
            return
            
        try:
            # Prepare data for Gemini
            team_stats = {
                self.team_home: {
                    'Possession (%)': round(self.team_stats[self.team_home]['Possession (%)'], 2),
                    'Distance (m)': round(self.team_stats[self.team_home]['Distance (m)'], 2),
                    'Passes': self.team_stats[self.team_home]['Passes'],
                    'Pass Completion (%)': self.team_stats[self.team_home].get('Pass Completion (%)', 0),
                    'Progressive Passes': self.team_stats[self.team_home].get('Progressive Passes', 0),
                    'Breaking Lines Passes': self.team_stats[self.team_home].get('Breaking Lines Passes', 0),
                    'Shots': self.team_stats[self.team_home]['Shots'],
                    'Shots on Target': self.team_stats[self.team_home].get('Shots on Target', 0),
                    'Shot Accuracy (%)': self.team_stats[self.team_home].get('Shot Accuracy (%)', 0),
                    'Expected Goals (xG)': self.team_stats[self.team_home].get('Expected Goals (xG)', 0),
                    'Formation': self.team_stats[self.team_home]['Most Used Formation'],
                    'Style': self.home_playstyle
                },
                self.team_away: {
                    'Possession (%)': round(self.team_stats[self.team_away]['Possession (%)'], 2),
                    'Distance (m)': round(self.team_stats[self.team_away]['Distance (m)'], 2),
                    'Passes': self.team_stats[self.team_away]['Passes'],
                    'Pass Completion (%)': self.team_stats[self.team_away].get('Pass Completion (%)', 0),
                    'Progressive Passes': self.team_stats[self.team_away].get('Progressive Passes', 0),
                    'Breaking Lines Passes': self.team_stats[self.team_away].get('Breaking Lines Passes', 0),
                    'Shots': self.team_stats[self.team_away]['Shots'],
                    'Shots on Target': self.team_stats[self.team_away].get('Shots on Target', 0),
                    'Shot Accuracy (%)': self.team_stats[self.team_away].get('Shot Accuracy (%)', 0),
                    'Expected Goals (xG)': self.team_stats[self.team_away].get('Expected Goals (xG)', 0),
                    'Formation': self.team_stats[self.team_away]['Most Used Formation'],
                    'Style': self.away_playstyle
                }
            }
            
            # Prepare pass data
            pass_stats = {
                self.team_home: {
                    'total': sum(1 for p in self.pass_data if p['team'] == self.team_home),
                    'forward': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'forward'),
                    'backward': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'backward'),
                    'lateral': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('direction') == 'lateral'),
                    'progressive': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('progressive', False)),
                    'breaking_lines': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('breaking_lines', False)),
                    'danger_zone': sum(1 for p in self.pass_data if p['team'] == self.team_home and p.get('danger_zone', False)),
                    'types': dict(self.pass_types[self.team_home])
                },
                self.team_away: {
                    'total': sum(1 for p in self.pass_data if p['team'] == self.team_away),
                    'forward': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'forward'),
                    'backward': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'backward'),
                    'lateral': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('direction') == 'lateral'),
                    'progressive': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('progressive', False)),
                    'breaking_lines': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('breaking_lines', False)),
                    'danger_zone': sum(1 for p in self.pass_data if p['team'] == self.team_away and p.get('danger_zone', False)),
                    'types': dict(self.pass_types[self.team_away])
                }
            }
            
            # Prepare shot data
            shot_stats = {
                self.team_home: {
                    'total': sum(1 for s in self.shot_data if s['team'] == self.team_home),
                    'on_target': sum(1 for s in self.shot_data if s['team'] == self.team_home and s.get('on_target', False)),
                    'xG': round(sum(s.get('expected_goal', 0) for s in self.shot_data if s['team'] == self.team_home), 3),
                    'under_pressure': self.shots_under_pressure[self.team_home],
                    'zones': dict(self.shot_zones[self.team_home]),
                    'types': dict(self.shot_types[self.team_home])
                },
                self.team_away: {
                    'total': sum(1 for s in self.shot_data if s['team'] == self.team_away),
                    'on_target': sum(1 for s in self.shot_data if s['team'] == self.team_away and s.get('on_target', False)),
                    'xG': round(sum(s.get('expected_goal', 0) for s in self.shot_data if s['team'] == self.team_away), 3),
                    'under_pressure': self.shots_under_pressure[self.team_away],
                    'zones': dict(self.shot_zones[self.team_away]),
                    'types': dict(self.shot_types[self.team_away])
                }
            }
            
            # Create a comprehensive prompt for Gemini
            prompt = f"""
            You are a professional football analyst with deep expertise in tactical analysis. 
            Analyze the following match data between {self.team_home} vs {self.team_away} and provide detailed tactical insights:
            
            # Match Information
            - Teams: {self.team_home} vs {self.team_away}
            - Home team playing style: {self.home_playstyle}
            - Away team playing style: {self.away_playstyle}
            
            # Team Stats
            {json.dumps(team_stats, indent=2)}
            
            # Pass Stats
            {json.dumps(pass_stats, indent=2)}
            
            # Shot Stats
            {json.dumps(shot_stats, indent=2)}
            
            # REQUIRED ANALYSIS
            Please provide ALL of the following sections:
            
            ## 1. Key Tactical Observations (5 points)
            Analyze the overall tactical approach of both teams based on the statistics. Focus on patterns, strategies, and key performance indicators.
            
            ## 2. Team Strengths - {self.team_home} (3 key points)
            Identify three specific strengths of {self.team_home} based on the data.
            
            ## 3. Team Weaknesses - {self.team_home} (3 key points)
            Identify three areas of improvement for {self.team_home} based on the data.
            
            ## 4. Team Strengths - {self.team_away} (3 key points)
            Identify three specific strengths of {self.team_away} based on the data.
            
            ## 5. Team Weaknesses - {self.team_away} (3 key points)
            Identify three areas of improvement for {self.team_away} based on the data.
            
            ## 6. Tactical Suggestions for {self.team_home} (5 specific points)
            Provide 5 detailed, specific tactical recommendations for {self.team_home} to exploit {self.team_away}'s weaknesses and counter their strengths.
            
            ## 7. Tactical Suggestions for {self.team_away} (5 specific points)
            Provide 5 detailed, specific tactical recommendations for {self.team_away} to exploit {self.team_home}'s weaknesses and counter their strengths.
            
            # Guidelines
            - Focus on tactical aspects, not just statistics
            - Provide specific, concrete observations and recommendations
            - Use football terminology and concepts
            - Explain your reasoning for each observation and suggestion
            - Be specific and detailed in your suggestions
            """
            
            # Make API call to Gemini
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 4096
                }
            }
            
            with self.st.spinner("Generating AI tactical insights..."):
                response = requests.post(
                    f"{api_url}?key={self.st.session_state.gemini_api_key}",
                    headers=headers,
                    json=data
                )
                
                # Process response
                if response.status_code == 200:
                    response_json = response.json()
                    
                    if 'candidates' in response_json and len(response_json['candidates']) > 0:
                        text_response = response_json['candidates'][0]['content']['parts'][0]['text']
                        
                        # Parse the response into different sections
                        sections = text_response.split("##")
                        
                        # Store insights
                        for section in sections:
                            # Skip empty sections
                            if not section.strip():
                                continue
                                
                            # Store observation in both teams' insights
                            if "Key Tactical Observations" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            # Store team-specific strengths and weaknesses
                            elif f"Team Strengths - {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                            elif f"Team Weaknesses - {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                            elif f"Team Strengths - {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            elif f"Team Weaknesses - {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                            # Store team-specific tactical suggestions
                            elif f"Tactical Suggestions for {self.team_home}" in section:
                                self.gemini_insights[self.team_home].append(f"## {section.strip()}")
                                # Extract suggestions for tactical suggestions
                                lines = section.strip().split("\n")[1:]  # Skip the heading
                                suggestions = []
                                for line in lines:
                                    line = line.strip()
                                    if line and (line[0].isdigit() or line[0] == '-'):
                                        suggestions.append(line[2:] if line[0] in ('1', '2', '3', '4', '5', '-') else line)
                                
                                if suggestions:
                                    self.tactical_suggestions[self.team_home] = suggestions[:5]
                            elif f"Tactical Suggestions for {self.team_away}" in section:
                                self.gemini_insights[self.team_away].append(f"## {section.strip()}")
                                # Extract suggestions for tactical suggestions
                                lines = section.strip().split("\n")[1:]  # Skip the heading
                                suggestions = []
                                for line in lines:
                                    line = line.strip()
                                    if line and (line[0].isdigit() or line[0] == '-'):
                                        suggestions.append(line[2:] if line[0] in ('1', '2', '3', '4', '5', '-') else line)
                                
                                if suggestions:
                                    self.tactical_suggestions[self.team_away] = suggestions[:5]
                        
                        self.st.success("✅ Generated AI insights successfully!")
                    else:
                        self.st.warning("Couldn't extract insights from Gemini response.")
                else:
                    self.st.error(f"Error calling Gemini API: {response.status_code} - {response.text}")
                    
        except Exception as e:
            self.st.error(f"Error generating AI insights: {str(e)}")
            traceback.print_exc()
            
    def run(self):
        """Main method to run the football analysis"""
        try:
            # Check if video info is available in session state
            if self.st.session_state[KEY_VIDEO_INFO]:
                self.video_info = self.st.session_state[KEY_VIDEO_INFO]
            
            # If analysis is already complete, just display the results
            if self.st.session_state[KEY_ANALYSIS_COMPLETE]:
                if self.st.session_state.processed_video_path and os.path.exists(self.st.session_state.processed_video_path):
                    self.display_video_analysis(self.st.session_state.processed_video_path)
                    self.display_player_stats()
                    self.display_spatial_analysis()
                    self.display_team_analysis()
                    
                    # Display strengths, weaknesses and tactical suggestions
                    if self.enable_tactical:
                        self.display_strengths_weaknesses()
                        self.display_tactical_suggestions()
                    
                    # Generate report if enabled
                    if self.enable_report:
                        self.generate_report()
                else:
                    self.st.error("Analysis results are incomplete or video file is missing. Please reset and try again.")
                    if self.st.button("Reset Analysis"):
                        for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                            self.st.session_state[key] = None
                        self.st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                        self.st.session_state[KEY_ANALYSIS_COMPLETE] = False
                        self.st.session_state.processed_video_path = None
                        self.st.rerun()
                return
                
            # If start analysis button is clicked and video is uploaded
            if self.start_analysis and self.video_path is not None:
                with self.st.spinner("Processing video and analyzing football match..."):
                    output_video_path = self.process_video()
                    
                if output_video_path and os.path.exists(output_video_path):
                    # Display results in tabs
                    self.display_video_analysis(output_video_path)
                    self.display_player_stats()
                    self.display_spatial_analysis()
                    self.display_team_analysis()
                    
                    # Display strengths, weaknesses and tactical suggestions
                    if self.enable_tactical:
                        self.display_strengths_weaknesses()
                        self.display_tactical_suggestions()
                    
                    # Generate report if enabled
                    if self.enable_report:
                        self.generate_report()
                    
                    self.st.sidebar.success("✅ Analysis completed successfully!")
                    
                    # Allow download of output video
                    with open(output_video_path, "rb") as file:
                        self.st.sidebar.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name="football_analysis_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    self.st.error("Video processing failed. Please check the logs and try again.")
            else:
                self.st.info("Upload a football match video and click 'Start Analysis' to begin.")
                
                # Show sample images of what the analysis will provide
                self.st.subheader("👁️ Preview of Analysis Capabilities")
                
                col1, col2, col3 = self.st.columns(3)
                
                with col1:
                    self.st.markdown("#### Enhanced Player Tracking")
                    self.st.markdown("""
                    The system tracks players across frames with:
                    - Team identification
                    - Position tracking
                    - Movement analysis
                    - Speed and acceleration metrics
                    - Pass network visualization
                    """)
                    
                with col2:
                    self.st.markdown("#### Advanced Analytics")
                    self.st.markdown("""
                    Advanced analysis features include:
                    - Heatmaps for player movement
                    - Enhanced pass and shot analysis
                    - Possession zones
                    - Advanced event detection
                    - Pass and shot quality metrics (xG, xA)
                    - Comprehensive team statistics
                    """)
                    
                with col3:
                    self.st.markdown("#### AI Tactical Analysis")
                    self.st.markdown("""
                    AI-powered tactical features:
                    - Team strengths & weaknesses detection
                    - Detailed tactical suggestions
                    - Opponent playstyle analysis
                    - Strategic recommendations
                    - SWOT analysis for each team
                    - Gemini AI integration for expert insights
                    """)
        except Exception as e:
            self.st.error(f"An error occurred: {str(e)}")
            traceback.print_exc()
            
            # Offer reset option
            if self.st.button("Reset Application"):
                for key in [KEY_ANALYSIS_COMPLETE, KEY_PLAYER_STATS, KEY_TEAM_STATS, KEY_EVENTS]:
                    self.st.session_state[key] = None
                self.st.session_state[KEY_VIDEO_INFO] = VideoInfo()
                self.st.session_state[KEY_ANALYSIS_COMPLETE] = False
                self.st.session_state.processed_video_path = None
                self.st.experimental_rerun()