import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.cluster import KMeans, DBSCAN
import os
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AI-FootballAnalyzer")

@dataclass
class Player:
    id: int
    position: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    team: Optional[str] = None
    role: Optional[str] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: List[Tuple[float, float]] = field(default_factory=list)
    conf: float = 0.0
    embedding: Optional[np.ndarray] = None

@dataclass
class Team:
    name: str
    players: List[Player] = field(default_factory=list)
    formation: str = "unknown"
    possession: float = 0.0
    style: str = "unknown"
    tactics: List[str] = field(default_factory=list)

@dataclass
class Match:
    home_team: Team
    away_team: Team
    ball_position: Optional[Tuple[float, float]] = None
    frame_count: int = 0
    events: List[Dict] = field(default_factory=list)

class PlayerEmbeddingNetwork(nn.Module):
    """Neural network to generate player embeddings from appearance and movement patterns"""
    def __init__(self, input_size=64, embedding_size=32):
        super(PlayerEmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (input_size//4) * (input_size//4), 128)
        self.fc2 = nn.Linear(128, embedding_size)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (x.size(2)) * (x.size(3)))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FormationClassifier(nn.Module):
    """Neural network to classify team formations based on player positions"""
    def __init__(self, input_size=22, num_formations=8):
        super(FormationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 64)  # x,y coordinates for each player
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_formations)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class TacticalAnalysisModel(nn.Module):
    """Neural network to generate tactical analysis from match state"""
    def __init__(self, input_features=64, num_tactics=12):
        super(TacticalAnalysisModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_tactics)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Probability for each tactical suggestion
        return x

class StyleClassifier(nn.Module):
    """Neural network to classify team playing style"""
    def __init__(self, input_features=32, num_styles=5):
        super(StyleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_styles)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AIFootballAnalyzer:
    """AI-driven football match analyzer"""
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.initialize_models()
        
        # Formation labels
        self.formation_labels = [
            "4-4-2", "4-3-3", "4-2-3-1", "3-5-2", "5-3-2", "3-4-3", "4-5-1", "4-1-4-1"
        ]
        
        # Playing style labels
        self.style_labels = [
            "Possession", "Counter-Attack", "High-Press", "Defensive", "Direct"
        ]
        
        # Tactical suggestion templates
        self.tactical_templates = {
            0: "Increase pressing intensity in the opponent's half",
            1: "Maintain compact defensive shape to limit passing options",
            2: "Use quick transitions to exploit space behind defensive line",
            3: "Focus on wing play to stretch opponent's defense",
            4: "Build play through central midfield areas",
            5: "Implement patient build-up from the back",
            6: "Apply counter-pressing immediately after losing possession",
            7: "Utilize direct passes to bypass pressing",
            8: "Position additional midfield support for defensive solidity",
            9: "Create overloads on the flanks for crossing opportunities",
            10: "Maintain high defensive line to compress playing space",
            11: "Employ low defensive block and focus on counter-attacks"
        }
        
        # Match state
        self.match = None
        self.frame_buffer = []
        self.feature_history = []
        
        # Player tracking params
        self.max_disappeared = 30
        self.similarity_threshold = 0.7
        
        # Team analysis memory
        self.team_analysis_memory = {
            'home': {style: 0 for style in self.style_labels},
            'away': {style: 0 for style in self.style_labels}
        }
    
    def initialize_models(self):
        """Initialize all neural network models"""
        # Player detection model (using YOLOv8)
        try:
            self.detection_model = YOLO("yolov8x.pt")
            logger.info("Loaded YOLOv8 detection model")
        except Exception as e:
            logger.error(f"Error loading detection model: {str(e)}")
            self.detection_model = None
        
        # Player embedding model
        self.embedding_model = PlayerEmbeddingNetwork().to(self.device)
        
        # Formation classification model
        self.formation_model = FormationClassifier(num_formations=8).to(self.device)
        
        # Tactical analysis model
        self.tactical_model = TacticalAnalysisModel(num_tactics=12).to(self.device)
        
        # Style classification model
        self.style_model = StyleClassifier(num_styles=5).to(self.device)
        
        # Try to load pre-trained weights if available
        try:
            self.embedding_model.load_state_dict(torch.load('models/embedding_model.pth', map_location=self.device))
            self.formation_model.load_state_dict(torch.load('models/formation_model.pth', map_location=self.device))
            self.tactical_model.load_state_dict(torch.load('models/tactical_model.pth', map_location=self.device))
            self.style_model.load_state_dict(torch.load('models/style_model.pth', map_location=self.device))
            logger.info("Loaded pre-trained AI models")
        except Exception as e:
            logger.warning(f"Could not load pre-trained models, using initialization weights: {str(e)}")
        
        # Set models to evaluation mode
        self.embedding_model.eval()
        self.formation_model.eval()
        self.tactical_model.eval()
        self.style_model.eval()
    
    def initialize_match(self, home_team_name="Home Team", away_team_name="Away Team"):
        """Initialize a new match analysis"""
        self.match = Match(
            home_team=Team(name=home_team_name),
            away_team=Team(name=away_team_name)
        )
        self.frame_buffer = []
        self.feature_history = []
        logger.info(f"Initialized match: {home_team_name} vs {away_team_name}")
    
    def detect_players(self, frame):
        """Detect players and ball in a frame using YOLOv8"""
        if self.detection_model is None:
            logger.error("Detection model not initialized")
            return [], None
        
        # Run YOLOv8 detection
        results = self.detection_model(frame)
        
        # Parse results
        players = []
        ball_position = None
        
        for i, det in enumerate(results[0].boxes):
            # Get bounding box
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf[0].cpu().numpy()
            cls_id = int(det.cls[0].cpu().numpy())
            
            # Calculate center position
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            if cls_id == 0:  # Person class in COCO dataset
                # Extract player patch for embedding
                player_patch = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Only process if patch is big enough
                if player_patch.shape[0] > 10 and player_patch.shape[1] > 10:
                    # Create player object
                    player = Player(
                        id=i,  # Temporary ID, will be updated during tracking
                        position=(center_x, center_y),
                        bbox=(x1, y1, x2, y2),
                        conf=conf
                    )
                    players.append(player)
            
            elif cls_id == 32:  # Sports ball class in COCO dataset
                ball_position = (center_x, center_y)
        
        return players, ball_position
    
    def generate_player_embeddings(self, frame, players):
        """Generate appearance embeddings for players"""
        if not players:
            return players
        
        # Transform for the embedding network
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            for player in players:
                x1, y1, x2, y2 = map(int, player.bbox)
                # Extract player patch
                patch = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    # Skip if invalid patch
                    player.embedding = np.zeros(32)
                    continue
                
                # Apply transform and get embedding
                try:
                    tensor = transform(patch).unsqueeze(0).to(self.device)
                    embedding = self.embedding_model(tensor).cpu().numpy()[0]
                    player.embedding = embedding
                except Exception as e:
                    logger.warning(f"Error generating embedding: {str(e)}")
                    player.embedding = np.zeros(32)
        
        return players
    
    def assign_teams_by_clustering(self, players):
        """Assign players to teams using clustering of appearance embeddings"""
        if not players or len(players) < 4:  # Need at least a few players from each team
            return players
        
        # Get player embeddings
        embeddings = np.array([p.embedding for p in players if p.embedding is not None])
        
        if len(embeddings) < 4:
            return players
        
        # Apply K-means clustering to separate teams
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Assign team labels based on clusters
        valid_players = [p for p in players if p.embedding is not None]
        for i, player in enumerate(valid_players):
            player.team = 'home' if clusters[i] == 0 else 'away'
        
        # Handle players without embeddings
        for player in players:
            if player.embedding is None:
                # Assign based on proximity to known team players
                home_distances = []
                away_distances = []
                
                for p in valid_players:
                    dist = np.sqrt((player.position[0] - p.position[0])**2 + 
                                   (player.position[1] - p.position[1])**2)
                    if p.team == 'home':
                        home_distances.append(dist)
                    else:
                        away_distances.append(dist)
                
                # Calculate average distance to each team
                avg_home = np.mean(home_distances) if home_distances else float('inf')
                avg_away = np.mean(away_distances) if away_distances else float('inf')
                
                # Assign to closest team
                player.team = 'home' if avg_home < avg_away else 'away'
        
        return players
    
    def track_players(self, prev_players, curr_players, frame_idx):
        """Track players between frames using appearance embeddings and Hungarian algorithm"""
        if not prev_players:
            # First frame, assign new IDs
            for i, player in enumerate(curr_players):
                player.id = i
            return curr_players
        
        # Build cost matrix based on embedding similarity and position
        cost_matrix = np.zeros((len(curr_players), len(prev_players)))
        
        for i, curr_player in enumerate(curr_players):
            for j, prev_player in enumerate(prev_players):
                # Calculate embedding similarity if available
                if curr_player.embedding is not None and prev_player.embedding is not None:
                    emb_similarity = np.dot(curr_player.embedding, prev_player.embedding) / (
                        np.linalg.norm(curr_player.embedding) * np.linalg.norm(prev_player.embedding))
                else:
                    emb_similarity = 0
                
                # Calculate position distance
                pos_distance = np.sqrt((curr_player.position[0] - prev_player.position[0])**2 + 
                                      (curr_player.position[1] - prev_player.position[1])**2)
                
                # Combined cost (lower is better)
                # Weight position more if embeddings aren't reliable
                if emb_similarity > 0:
                    cost = (1 - emb_similarity) * 0.6 + (pos_distance / 100) * 0.4
                else:
                    cost = pos_distance / 100
                
                # Add team mismatch penalty
                if curr_player.team is not None and prev_player.team is not None and curr_player.team != prev_player.team:
                    cost += 100  # High penalty for team mismatch
                
                cost_matrix[i, j] = cost
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Update player IDs and history
        new_players = []
        for i, curr_idx in enumerate(row_indices):
            prev_idx = col_indices[i]
            curr_player = curr_players[curr_idx]
            prev_player = prev_players[prev_idx]
            
            # Only match if cost is reasonable
            if cost_matrix[curr_idx, prev_idx] < 10:  # Threshold for match
                # Transfer ID and history
                curr_player.id = prev_player.id
                curr_player.history = prev_player.history.copy()
                
                # Calculate velocity
                dt = 1.0  # Assume constant time step
                curr_player.velocity = (
                    (curr_player.position[0] - prev_player.position[0]) / dt,
                    (curr_player.position[1] - prev_player.position[1]) / dt
                )
                
                # If team was assigned in previous frame but not current, maintain it
                if curr_player.team is None and prev_player.team is not None:
                    curr_player.team = prev_player.team
                
                # If role was assigned in previous frame, maintain it
                if prev_player.role is not None:
                    curr_player.role = prev_player.role
            else:
                # Cost too high, treat as new player
                curr_player.id = max([p.id for p in prev_players], default=-1) + 1
                curr_player.history = []
                curr_player.velocity = (0, 0)
            
            # Add current position to history
            curr_player.history.append(curr_player.position)
            
            # Limit history length
            if len(curr_player.history) > 30:
                curr_player.history.pop(0)
            
            new_players.append(curr_player)
        
        # Handle unmatched detections
        matched_indices = set(row_indices)
        for i in range(len(curr_players)):
            if i not in matched_indices:
                curr_player = curr_players[i]
                curr_player.id = max([p.id for p in new_players], default=-1) + 1
                curr_player.history = [curr_player.position]
                curr_player.velocity = (0, 0)
                new_players.append(curr_player)
        
        return new_players
    
    def assign_player_roles(self, players):
        """Assign roles to players based on position and movement patterns"""
        if not players:
            return players
        
        # Separate players by team
        home_players = [p for p in players if p.team == 'home']
        away_players = [p for p in players if p.team == 'away']
        
        # Process each team
        for team_players in [home_players, away_players]:
            if not team_players:
                continue
            
            # Find goalkeeper - usually the player furthest back for each team
            y_positions = [p.position[1] for p in team_players]
            max_y_idx = np.argmax(y_positions)  # Assuming y increases downward
            team_players[max_y_idx].role = 'goalkeeper'
            
            # Get outfield players
            outfield_players = [p for i, p in enumerate(team_players) if i != max_y_idx]
            
            # Sort by y position
            outfield_players.sort(key=lambda p: p.position[1])
            num_players = len(outfield_players)
            
            # Divide into defenders, midfielders and attackers based on position
            num_defenders = max(1, num_players // 3)
            num_midfielders = max(1, num_players // 3)
            num_attackers = num_players - num_defenders - num_midfielders
            
            # Assign roles
            for i, player in enumerate(outfield_players):
                if i < num_attackers:
                    player.role = 'attacker'
                elif i < num_attackers + num_midfielders:
                    player.role = 'midfielder'
                else:
                    player.role = 'defender'
            
            # Try to identify wide players based on x position
            for role in ['attacker', 'midfielder', 'defender']:
                role_players = [p for p in outfield_players if p.role == role]
                if len(role_players) >= 3:
                    # Sort by x position
                    role_players.sort(key=lambda p: p.position[0])
                    # Leftmost is left, rightmost is right
                    role_players[0].role = f'left_{role}'
                    role_players[-1].role = f'right_{role}'
        
        return players
    
    def detect_formation(self, players):
        """Detect team formation using the neural network model"""
        if not players:
            return "Unknown"
        
        # Need at least 7 players for meaningful formation detection
        if len(players) < 7:
            return "Unknown"
        
        # Get player positions normalized to 0-1 range
        # Find field bounds
        min_x = min(p.position[0] for p in players)
        max_x = max(p.position[0] for p in players)
        min_y = min(p.position[1] for p in players)
        max_y = max(p.position[1] for p in players)
        
        # Normalize positions
        positions = []
        for player in players:
            if player.role != 'goalkeeper':  # Exclude goalkeeper from formation
                norm_x = (player.position[0] - min_x) / (max_x - min_x) if max_x > min_x else 0.5
                norm_y = (player.position[1] - min_y) / (max_y - min_y) if max_y > min_y else 0.5
                positions.append(norm_x)
                positions.append(norm_y)
        
        # Pad or truncate to fixed length (20 coordinates for 10 outfield players)
        while len(positions) < 20:
            positions.append(0.5)  # Default position
        positions = positions[:20]  # Truncate if too many
        
        # Convert to tensor for model input
        with torch.no_grad():
            input_tensor = torch.tensor(positions, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.formation_model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            
            # Get formation label
            formation = self.formation_labels[predicted_idx]
            
            # Calculate confidence
            confidence = torch.nn.functional.softmax(output, dim=1)[0, predicted_idx].item()
            
            # Only return if confidence is high enough
            if confidence > 0.5:
                return formation
            else:
                # Fallback to heuristic approach for low confidence
                return self.detect_formation_heuristic(players)
    
    def detect_formation_heuristic(self, players):
        """Fallback heuristic method for formation detection"""
        # Filter out goalkeepers
        outfield_players = [p for p in players if p.role != 'goalkeeper']
        
        if not outfield_players:
            return "Unknown"
        
        # Count players in each role
        defenders = sum(1 for p in outfield_players if p.role in ['defender', 'left_defender', 'right_defender'])
        midfielders = sum(1 for p in outfield_players if p.role in ['midfielder', 'left_midfielder', 'right_midfielder'])
        attackers = sum(1 for p in outfield_players if p.role in ['attacker', 'left_attacker', 'right_attacker'])
        
        # Build formation string
        formation = f"{defenders}-{midfielders}-{attackers}"
        
        # Check for special formations
        if defenders == 4 and midfielders == 2 and attackers == 3:
            formation = "4-2-3-1"  # Special case for 4-2-3-1
        elif defenders == 4 and midfielders == 1 and attackers == 4:
            formation = "4-1-4-1"  # Special case for 4-1-4-1
        
        return formation
    
    def analyze_team_style(self, team_features):
        """Analyze team playing style using the style classification model"""
        with torch.no_grad():
            # Normalize features
            features = np.array(team_features)
            if np.std(features) > 0:
                features = (features - np.mean(features)) / np.std(features)
            
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get style probabilities
            output = self.style_model(input_tensor)
            style_probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            
            # Update style memory with exponential smoothing
            alpha = 0.1  # Smoothing factor
            for i, prob in enumerate(style_probs):
                self.team_analysis_memory[team_features['team']][self.style_labels[i]] = (
                    (1 - alpha) * self.team_analysis_memory[team_features['team']][self.style_labels[i]] + 
                    alpha * prob
                )
            
            # Return dominant style
            dominant_style_idx = np.argmax([
                self.team_analysis_memory[team_features['team']][style] 
                for style in self.style_labels
            ])
            
            return self.style_labels[dominant_style_idx]
    
    def generate_tactical_suggestions(self, team_data, opponent_data):
        """Generate tactical suggestions using the tactical analysis model"""
        # Combine team and opponent data for context
        features = np.concatenate([
            team_data['features'],
            opponent_data['features'],
            [team_data['possession'] / 100]  # Possession as a feature
        ])
        
        # Normalize features
        if np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)
        
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get tactical suggestion probabilities
            output = self.tactical_model(input_tensor).cpu().numpy()[0]
            
            # Select tactics above threshold
            tactics_indices = np.where(output > 0.5)[0]
            
            # If no tactics above threshold, select top 3
            if len(tactics_indices) == 0:
                tactics_indices = np.argsort(output)[-3:]
            
            # Convert to tactical suggestions
            suggestions = [self.tactical_templates[idx] for idx in tactics_indices]
            
            return suggestions
    
    def extract_team_features(self, team_players, frame_width, frame_height):
        """Extract team features for style and tactical analysis"""
        if not team_players:
            return np.zeros(32)
        
        # Calculate team features
        features = []
        
        # Calculate team centroid
        centroid_x = np.mean([p.position[0] for p in team_players]) / frame_width
        centroid_y = np.mean([p.position[1] for p in team_players]) / frame_height
        features.extend([centroid_x, centroid_y])
        
        # Calculate team width and height
        width = (max(p.position[0] for p in team_players) - 
                min(p.position[0] for p in team_players)) / frame_width
        height = (max(p.position[1] for p in team_players) - 
                min(p.position[1] for p in team_players)) / frame_height
        features.extend([width, height])
        
        # Calculate average velocity magnitude
        avg_vel_mag = np.mean([np.sqrt(p.velocity[0]**2 + p.velocity[1]**2) 
                              for p in team_players]) / 10.0  # Normalize
        features.append(avg_vel_mag)
        
        # Calculate team density (average distance to centroid)
        avg_dist_to_centroid = np.mean([
            np.sqrt((p.position[0]/frame_width - centroid_x)**2 + 
                    (p.position[1]/frame_height - centroid_y)**2)
            for p in team_players
        ])
        features.append(avg_dist_to_centroid)
        
        # Calculate role distribution
        role_counts = {
            'defender': sum(1 for p in team_players if p.role in ['defender', 'left_defender', 'right_defender']),
            'midfielder': sum(1 for p in team_players if p.role in ['midfielder', 'left_midfielder', 'right_midfielder']),
            'attacker': sum(1 for p in team_players if p.role in ['attacker', 'left_attacker', 'right_attacker'])
        }
        total = max(sum(role_counts.values()), 1)
        features.extend([count / total for count in role_counts.values()])
        
        # Pad or truncate to ensure consistent length
        while len(features) < 32:
            features.append(0)
        features = features[:32]
        
        return np.array(features)
    
    def analyze_frame(self, frame, frame_idx):
        """Process a single video frame for analysis"""
        # Initialize match if necessary
        if self.match is None:
            self.initialize_match()
        
        # Detect players and ball
        players, ball_position = self.detect_players(frame)
        
        # Generate player embeddings
        players = self.generate_player_embeddings(frame, players)
        
        # Assign teams
        players = self.assign_teams_by_clustering(players)
        
        # Track players across frames
        if hasattr(self, 'prev_players') and self.prev_players:
            players = self.track_players(self.prev_players, players, frame_idx)
        self.prev_players = players.copy()
        
        # Assign player roles
        players = self.assign_player_roles(players)
        
        # Update match state
        self.match.ball_position = ball_position
        self.match.frame_count = frame_idx
        
        # Separate players by team
        home_players = [p for p in players if p.team == 'home']
        away_players = [p for p in players if p.team == 'away']
        
        self.match.home_team.players = home_players
        self.match.away_team.players = away_players
        
        # Extract team features
        frame_height, frame_width = frame.shape[:2]
        home_features = self.extract_team_features(home_players, frame_width, frame_height)
        away_features = self.extract_team_features(away_players, frame_width, frame_height)
        
        # Store features for later analysis
        self.feature_history.append({
            'frame_idx': frame_idx,
            'home_features': home_features,
            'away_features': away_features,
            'ball_position': ball_position
        })
        
        # Limit feature history length
        if len(self.feature_history) > 100:
            self.feature_history.pop(0)
        
        # Only run formation and style analysis periodically to save processing
        if frame_idx % 30 == 0 and len(self.feature_history) >= 30:
            # Detect formations
            self.match.home_team.formation = self.detect_formation(home_players)
            self.match.away_team.formation = self.detect_formation(away_players)
            
            # Calculate possession
            ball_possession = {'home': 0, 'away': 0}
            for i in range(min(30, len(self.feature_history))):
                if self.feature_history[-(i+1)]['ball_position'] is not None:
                    ball_pos = self.feature_history[-(i+1)]['ball_position']
                    home_centroid = np.mean([p.position for p in self.match.home_team.players], axis=0) if self.match.home_team.players else None
                    away_centroid = np.mean([p.position for p in self.match.away_team.players], axis=0) if self.match.away_team.players else None
                    
                    if home_centroid is not None and away_centroid is not None:
                        home_dist = np.sqrt((ball_pos[0] - home_centroid[0])**2 + (ball_pos[1] - home_centroid[1])**2)
                        away_dist = np.sqrt((ball_pos[0] - away_centroid[0])**2 + (ball_pos[1] - away_centroid[1])**2)
                        
                        if home_dist < away_dist:
                            ball_possession['home'] += 1
                        else:
                            ball_possession['away'] += 1
            
            total_possession = ball_possession['home'] + ball_possession['away']
            if total_possession > 0:
                self.match.home_team.possession = ball_possession['home'] / total_possession * 100
                self.match.away_team.possession = ball_possession['away'] / total_possession * 100
            
            # Analyze team styles using last 30 frames of features
            home_style_features = {
                'team': 'home',
                'features': np.mean([f['home_features'] for f in self.feature_history[-30:]], axis=0)
            }
            away_style_features = {
                'team': 'away',
                'features': np.mean([f['away_features'] for f in self.feature_history[-30:]], axis=0)
            }
            
            self.match.home_team.style = self.analyze_team_style(home_style_features)
            self.match.away_team.style = self.analyze_team_style(away_style_features)
            
            # Generate tactical suggestions
            home_tactical_data = {
                'features': home_style_features['features'],
                'possession': self.match.home_team.possession
            }
            away_tactical_data = {
                'features': away_style_features['features'],
                'possession': self.match.away_team.possession
            }
            
            self.match.home_team.tactics = self.generate_tactical_suggestions(
                home_tactical_data, away_tactical_data
            )
            self.match.away_team.tactics = self.generate_tactical_suggestions(
                away_tactical_data, home_tactical_data
            )
            
        # Visualize analysis results on frame
        return self.visualize_analysis(frame, players, ball_position)
    
    def visualize_analysis(self, frame, players, ball_position):
        """Visualize analysis results on frame"""
        # Draw player tracking
        for player in players:
            # Determine color based on team
            if player.team == 'home':
                color = (0, 0, 255)  # Red for home team
            elif player.team == 'away':
                color = (255, 0, 0)  # Blue for away team
            else:
                color = (200, 200, 200)  # Gray for unknown
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, player.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and role
            role_label = player.role[:3] if player.role else '?'
            label = f"{player.id} ({role_label})"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw motion history if available
            if len(player.history) > 1:
                history = np.array(player.history)
                for i in range(1, len(history)):
                    start = tuple(map(int, history[i-1]))
                    end = tuple(map(int, history[i]))
                    cv2.line(frame, start, end, color, 1)
        
        # Draw ball position
        if ball_position:
            cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), 10, (0, 255, 255), -1)
        
        # Draw match information
        cv2.putText(frame, f"Frame: {self.match.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.match:
            # Draw team names, formations, and possession
            home_info = f"{self.match.home_team.name} ({self.match.home_team.formation})"
            away_info = f"{self.match.away_team.name} ({self.match.away_team.formation})"
            
            cv2.putText(frame, home_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, away_info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw possession stats if available
            if hasattr(self.match.home_team, 'possession') and hasattr(self.match.away_team, 'possession'):
                possession_info = f"Possession: {self.match.home_team.possession:.1f}% - {self.match.away_team.possession:.1f}%"
                cv2.putText(frame, possession_info, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw playing styles
            home_style = f"Style: {self.match.home_team.style}" if hasattr(self.match.home_team, 'style') else ""
            away_style = f"Style: {self.match.away_team.style}" if hasattr(self.match.away_team, 'style') else ""
            
            cv2.putText(frame, home_style, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, away_style, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, max_frames=None, progress_callback=None):
        """Process a full video file"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is None:
            max_frames = total_frames
        else:
            max_frames = min(max_frames, total_frames)
        
        # Initialize match
        home_team = os.path.basename(video_path).split('_')[0] if '_' in os.path.basename(video_path) else "Home Team"
        away_team = os.path.basename(video_path).split('_')[1].split('.')[0] if '_' in os.path.basename(video_path) else "Away Team"
        self.initialize_match(home_team, away_team)
        
        # Initialize video writer
        if output_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(video_path)}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        frame_idx = 0
        frame_skip = 3  # Process every nth frame for efficiency
        
        logger.info(f"Starting video processing: {video_path}")
        start_time = time.time()
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_idx % frame_skip == 0:
                # Process frame
                processed_frame = self.analyze_frame(frame, frame_idx)
                
                # Write processed frame
                out.write(processed_frame)
                
                # Update progress
                if progress_callback:
                    progress = frame_count / max_frames
                    progress_callback(progress, frame_count, max_frames)
                
                # Log progress
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {frame_count}/{max_frames} frames ({fps_processing:.1f} fps)")
                
                frame_count += 1
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        logger.info(f"Video processing complete: {output_path}")
        
        # Generate analysis report
        self.generate_report(os.path.dirname(output_path))
        
        return output_path
    
    def generate_report(self, output_dir):
        """Generate analysis report with visualizations"""
        if self.match is None:
            logger.error("No match data available for report generation")
            return
        
        # Create report directory
        report_dir = os.path.join(output_dir, "analysis_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate HTML report
        report_path = os.path.join(report_dir, "tactical_report.html")
        
        with open(report_path, "w") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Football Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                    h1, h2, h3 {{ color: #0066cc; }}
                    .container {{ display: flex; justify-content: space-between; }}
                    .team-section {{ width: 48%; padding: 15px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }}
                    .home {{ border-left: 4px solid #cc0000; }}
                    .away {{ border-left: 4px solid #0000cc; }}
                    .tactics-list {{ background-color: #f9f9f9; padding: 10px; border-radius: 5px; }}
                    .tactic-item {{ margin-bottom: 8px; padding: 8px; background-color: #f0f0f0; border-radius: 3px; }}
                    .stats-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                    .stats-table th {{ background-color: #e0e0e0; padding: 8px; text-align: left; }}
                    .stats-table td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                    .formation-visual {{ text-align: center; margin: 15px 0; }}
                </style>
            </head>
            <body>
                <h1>AI Football Analysis Report</h1>
                <h2>{self.match.home_team.name} vs {self.match.away_team.name}</h2>
                
                <div class="container">
                    <div class="team-section home">
                        <h3>{self.match.home_team.name}</h3>
                        <p><strong>Formation:</strong> {self.match.home_team.formation}</p>
                        <p><strong>Playing Style:</strong> {self.match.home_team.style}</p>
                        <p><strong>Possession:</strong> {self.match.home_team.possession:.1f}%</p>
                        
                        <h4>AI-Generated Tactical Recommendations</h4>
                        <div class="tactics-list">
                            {''.join(f'<div class="tactic-item">{tactic}</div>' for tactic in self.match.home_team.tactics)}
                        </div>
                    </div>
                    
                    <div class="team-section away">
                        <h3>{self.match.away_team.name}</h3>
                        <p><strong>Formation:</strong> {self.match.away_team.formation}</p>
                        <p><strong>Playing Style:</strong> {self.match.away_team.style}</p>
                        <p><strong>Possession:</strong> {self.match.away_team.possession:.1f}%</p>
                        
                        <h4>AI-Generated Tactical Recommendations</h4>
                        <div class="tactics-list">
                            {''.join(f'<div class="tactic-item">{tactic}</div>' for tactic in self.match.away_team.tactics)}
                        </div>
                    </div>
                </div>
                
                <h3>Match Analysis</h3>
                <p>This analysis was generated by an AI system that detects formations, playing styles, and generates 
                tactical recommendations based on real-time video analysis.</p>
                
                <h4>Key Tactical Observations</h4>
                <ul>
                    <li>The match features a {self.match.home_team.formation} formation from {self.match.home_team.name} 
                    against a {self.match.away_team.formation} formation from {self.match.away_team.name}.</li>
                    <li>{self.match.home_team.name} employed a {self.match.home_team.style} playing style,
                    while {self.match.away_team.name} showed a {self.match.away_team.style} approach.</li>
                    <li>Possession was {self.match.home_team.possession:.1f}% for {self.match.home_team.name} and 
                    {self.match.away_team.possession:.1f}% for {self.match.away_team.name}.</li>
                </ul>
                
                <p><i>Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</i></p>
            </body>
            </html>
            """)
        
        logger.info(f"Analysis report generated: {report_path}")
        return report_path

# For testing independently
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="AI Football Analyzer")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--max_frames", type=int, default=1000, help="Maximum frames to process")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AIFootballAnalyzer(use_gpu=args.gpu)
    
    # Process video
    output_path = analyzer.process_video(args.video, args.output, args.max_frames)
    
    print(f"Analysis complete. Output saved to: {output_path}")