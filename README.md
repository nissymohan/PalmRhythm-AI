# 🎵 PalmRhythm: Hand-Gesture Controlled Music System

An intelligent, real-time music generation system controlled entirely by hand gestures. PalmRhythm combines computer vision, musical theory, and adaptive learning to create a personalized musical experience that responds to your movements.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![JavaScript](https://img.shields.io/badge/javascript-ES6+-yellow.svg)

## 🌟 Features

### 🎹 Intelligent Pattern Generation
- **Context-aware music creation** using probabilistic models and music theory
- **5 musical modes**: Happy (Major), Calm (Natural Minor), Energetic (Lydian), Mysterious (Phrygian), and Bollywood (Bhairav-inspired)
- **Markov chain-based** note transitions for natural musical flow
- **Real-time generation** with <10ms latency

### 🤖 Adaptive Learning System
- **Personalized experience** that learns your playing style over time
- **Preference tracking** for favorite notes, tempos, and musical styles
- **Statistical analysis** of gesture patterns
- **Confidence metrics** that improve with usage
- **Learning visualization** with progress curves and heatmaps

### 👋 Gesture Recognition
- **Hand tracking** using MediaPipe and TensorFlow.js
- **Multi-parameter mapping**:
  - Hand height → Pitch range
  - Velocity → Rhythm complexity
  - Curvature → Pattern smoothness
  - Palm distance → Note count
  - Finger states → Musical style selection

### 🎨 Visual Feedback
- Real-time hand skeleton overlay
- Learning progress indicators
- Style and complexity displays
- Performance metrics

## 🚀 Getting Started

### Prerequisites

- Modern web browser (Chrome recommended for best compatibility)
- Webcam
- Python 3.x (for local server)

### Installation

1. **Clone or download the project files**
   ```bash
   git clone https://github.com/nissymohan/PalmRhythm-AI.git
   cd PalmRhythm-AI
   ```

2. **Verify you have these files**:
   - `index.html` - Main application
   - `IntelligentPatternGenerator.js` - Music generation engine
   - `AdaptiveLearning.js` - Learning system
   - `camera-test.html` - Camera diagnostics tool

### Running the Application

**Important**: The application requires a local server due to browser security restrictions for camera access.

1. **Start a local server**:
   ```bash
   # Using Python 3
   python -m http.server 8000
   
   # Or Python 2
   python -m SimpleHTTPServer 8000
   ```

2. **Open in browser**:
   ```
   http://localhost:8000/index.html
   ```

3. **Allow camera access** when prompted

### Testing Camera Access

If you encounter camera issues, use the diagnostic tool:

```
http://localhost:8000/camera-test.html
```

This tool will help identify and resolve common camera permission problems.

## 🎮 How to Use

### Basic Controls

1. **Start the system**: Click the start button and allow camera access
2. **Position yourself**: Ensure your hands are visible in the camera frame
3. **Make gestures**: Move your hands to control the music
4. **Select styles**: Use different finger combinations to switch musical modes

### Gesture Mapping

| Gesture | Effect |
|---------|--------|
| **Hand Height** | Controls pitch (higher hand = higher notes) |
| **Movement Speed** | Affects rhythm complexity and energy |
| **Hand Smoothness** | Determines melodic step size |
| **Hand Separation** | Controls number of notes in pattern |
| **0 Fingers Up** | Calm style (Natural Minor) |
| **1 Finger Up** | Happy style (Major scale) |
| **2 Fingers Up** | Energetic style (Lydian mode) |
| **3 Fingers Up** | Mysterious style (Phrygian mode) |
| **4-5 Fingers Up** | Bollywood style (Bhairav-inspired) |

### Learning System

The system learns from your playing:

- **Patterns 0-5**: Initializing
- **Patterns 5-20**: Learning your style
- **Patterns 20-50**: Adapting to preferences
- **Patterns 50+**: Fully personalized

View your learning progress in real-time through the on-screen metrics.

## 🏗️ Architecture

### Core Components

#### 1. IntelligentPatternGenerator
The music generation engine that creates melodic sequences:

- **Probabilistic note selection** using Markov chains
- **Music theory integration** with proper scale degrees
- **Context-aware generation** based on gesture analysis
- **Adaptive transition matrices** that learn from played patterns

```javascript
// Example usage
const generator = new IntelligentPatternGenerator();
const pattern = generator.generatePattern({
    handHeight: 0.7,
    velocity: 0.5,
    curvature: 0.6,
    palmDistance: 0.5,
    fingerStates: { thumb: true, index: true }
});
```

#### 2. AdaptiveLearning
Tracks and learns user preferences:

- **Preference tracking** for notes, styles, and patterns
- **Statistical analysis** of playing behavior
- **Confidence metrics** based on pattern count
- **Visualization data** for learning curves and heatmaps

```javascript
// Example usage
const learning = new AdaptiveLearning();
learning.recordPattern(pattern, gestureFeatures, 'happy');
const stats = learning.getStats();
```

#### 3. Gesture Recognition
Computer vision system using MediaPipe:

- Real-time hand landmark detection
- 21-point hand skeleton tracking
- Feature extraction for musical mapping
- Multi-hand support

### Data Flow

```
Camera Feed → MediaPipe → Gesture Features → Pattern Generator → MIDI Notes → Audio
                                    ↓
                            Adaptive Learning
                                    ↓
                            Preference Update
```

## 🎼 Musical Theory

### Scale Modes

The system uses authentic musical modes for different moods:

- **Major (Happy)**: Bright, uplifting sound
- **Natural Minor (Calm)**: Relaxed, contemplative
- **Lydian (Energetic)**: Bright with raised 4th, floating quality
- **Phrygian (Mysterious)**: Dark with flattened 2nd, exotic feel
- **Bhairav (Bollywood)**: Indian classical-inspired scale

### Transition Probabilities

Notes transition based on music theory principles:
- Strong tendency toward tonic (root note)
- Dominants resolve to tonics
- Stepwise motion preferred over large leaps
- Energy level modulates jump probability

## 📊 Performance Metrics

### System Performance
- Pattern generation latency: <10ms
- Hand tracking: 30 FPS
- Audio synthesis: Web Audio API (real-time)

### Learning Metrics
- Learning progress: 0-100% (based on pattern count)
- Confidence: 0-100% (increases with consistency)
- Adaptation: 0-100% (measures personalization level)

## 🛠️ Troubleshooting

### Camera Issues

**Problem**: Camera access denied
- **Solution**: Click the padlock icon in the URL bar and reset camera permissions

**Problem**: "file://" protocol detected
- **Solution**: Must use `http://localhost` - see installation instructions

**Problem**: Camera already in use
- **Solution**: Close other applications (Zoom, Teams, Skype) using the camera

**Problem**: No video appears
- **Solution**: Check browser console for errors, ensure adequate lighting

### Performance Issues

**Problem**: Laggy or slow response
- **Solution**: Close other browser tabs, ensure good lighting, try Chrome browser

**Problem**: Audio glitches
- **Solution**: Reduce complexity setting, check CPU usage

### Learning System

**Problem**: System not adapting
- **Solution**: Play at least 10 patterns for initial learning, 50+ for full personalization

## 🔬 Technical Details

### Dependencies
- **TensorFlow.js**: Machine learning framework
- **MediaPipe Hands**: Hand tracking model
- **Web Audio API**: Real-time audio synthesis
- **ES6 Modules**: Modern JavaScript architecture

### Browser Compatibility
- ✅ Chrome 90+ (Recommended)
- ✅ Firefox 88+
- ✅ Edge 90+
- ⚠️ Safari 14+ (limited support)

### System Requirements
- Webcam (720p or higher recommended)
- 4GB RAM minimum
- Modern CPU (i5/Ryzen 5 or better)
- Well-lit environment

## 📈 Future Enhancements

- [ ] MIDI output support for external instruments
- [ ] Recording and playback functionality
- [ ] Multiple simultaneous players
- [ ] Advanced rhythm patterns
- [ ] Custom scale editor
- [ ] Cloud-based preference sync
- [ ] Mobile app version
- [ ] VR/AR integration

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional musical scales and modes
- Enhanced gesture recognition
- Performance optimizations
- UI/UX improvements
- Documentation and tutorials

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Vinayak**  
Researcher, IIT Kanpur  
Goswami Group

Developed as part of AI/ML course submission exploring real-time human-computer interaction through gesture-based musical interfaces.

## 🙏 Acknowledgments

- MediaPipe team for hand tracking technology
- TensorFlow.js community
- Music theory inspiration from classical and Indian classical traditions
- IIT Kanpur AI/ML course instructors

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section above
2. Review browser console for error messages
3. Ensure all prerequisites are met
4. Test with `camera-test.html` diagnostic tool

---

**Note**: This is an educational project demonstrating the integration of computer vision, adaptive learning, and generative music systems. Performance may vary based on hardware and environmental conditions.

**Made with ❤️ and JavaScript**
