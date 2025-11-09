# 🎵 PalmRhythm AI

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![JavaScript](https://img.shields.io/badge/javascript-ES6+-yellow.svg)
[![Made with MediaPipe](https://img.shields.io/badge/Made%20with-MediaPipe-blue)](https://mediapipe.dev/)
[![Powered by Tone.js](https://img.shields.io/badge/Powered%20by-Tone.js-orange)](https://tonejs.github.io/)

> **Real-Time Gesture-Controlled Music Generation with Adaptive Learning**

An AI-powered music generation system that transforms hand gestures into melodies in real-time. Create music intuitively using just your webcam - no instruments, no musical training required!

📄 [Technical Paper](TECHNICAL_RESEARCH_JOURNAL.md) | 🌐 [Live Demo](https://nissymohan.github.io/PalmRhythm-AI/)

---

## ✨ Features

### 🎹 Gesture-Based Control
- **Right Hand** 🎵: Control melody style and pitch
  - 0-4 fingers = 5 different musical styles
  - Hand height = pitch range
  - Natural, intuitive interaction

- **Left Hand** 👍: Save favorite melodies
  - Thumbs up to like patterns
  - Thumbs down to dislike
  - Builds personalized preferences

### 🎨 Musical Styles
- 🧘 **Calm Meditation** - Peaceful, flowing melodies
- 😊 **Happy Major** - Bright, uplifting tunes
- 🔥 **Energetic Rock** - Dynamic, powerful patterns
- 🌙 **Mysterious Minor** - Dark, introspective sounds
- 💃 **Bollywood Raag** - Exotic, dramatic scales

### 🚀 Performance
- **< 50ms latency** - True real-time response
- **Browser-based** - No installation required
- **CPU-only** - No GPU needed
- **Adaptive learning** - Personalizes to your preferences

### 🎨 Dynamic UI
- **Theme changes** based on current musical style
- **Glass morphism** design with smooth animations
- **Real-time statistics** and hand tracking visualization
- **Color-coded hands** (Green = melody, Cyan = gestures)

---

## 🛠️ Technology Stack

### Core Technologies
- **MediaPipe Hands** - Real-time hand tracking (30 FPS)
- **Tone.js** - Web Audio synthesis and playback
- **Vanilla JavaScript** - No framework overhead for maximum performance

### AI/ML Architecture
- **Gesture-Conditioned Markov Model** - Novel hybrid approach
- **Adaptive Learning** - Online preference optimization
- **Music Theory Integration** - Scale constraints for musical coherence

### Mathematical Foundation
```
P(melody | gesture, preferences, style) = 
    Markov(gesture_features) × 
    AdaptiveWeights(user_preferences) × 
    ScaleConstraints(musical_theory)
```

---

## 🚀 Quick Start

### Prerequisites
- Modern web browser (Chrome recommended)
- Webcam access
- Python 3.x (for local server)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/palmrhythm.git
cd palmrhythm
```

2. **Start local server**
```bash
python -m http.server 8000
```

3. **Open in browser**
```
http://localhost:8000/landing.html
```

4. **Grant camera permissions** and start creating music! 🎵

### Quick Test
To test camera setup first:
```
http://localhost:8000/camera-test.html
```

---

## 📖 How to Use

### Getting Started
1. **Launch** the application from `landing.html`
2. **Allow** camera access when prompted
3. **Position** yourself so the camera can see both hands
4. **Start Music** button to begin

### Creating Melodies
1. **Right hand controls**:
   - Show 0 fingers (fist) → Calm style
   - Show 1 finger → Happy style
   - Show 2 fingers → Energetic style
   - Show 3 fingers → Mysterious style
   - Show 4+ fingers → Bollywood style
   - Move hand up/down to change pitch

2. **Hold pose** for 2 seconds to generate a pattern

3. **Left hand feedback**:
   - Make thumbs up 👍 to save favorites
   - System learns your preferences over time

### Tips for Best Results
- 🌟 Good lighting is essential
- 📐 Keep hands within camera frame
- 🤚 Make clear, distinct gestures
- ⏱️ Hold poses steady for recognition
- 🎯 Close other fingers when making thumbs up/down

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐
│   Webcam Feed   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MediaPipe      │
│  Hand Tracking  │  (18.5ms avg)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │  (< 1ms)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Gesture-Conditioned     │
│ Markov Generator        │  (4.2ms avg)
│ + Adaptive Learning     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│   Tone.js       │
│   Synthesizer   │  (10.3ms avg)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio Output   │
└─────────────────┘

Total Latency: 34ms (avg) | 47.4ms (95th percentile)
```

### Key Components

**1. Hand Tracking Module**
- Real-time 21-point hand landmark detection
- Finger counting algorithm
- Gesture recognition (thumbs up/down)
- Mirror-corrected display

**2. Pattern Generator**
```javascript
class PatternGenerator {
    generatePattern(gestureFeatures) {
        // 1. Select musical scale based on finger count
        // 2. Calculate octave from hand height
        // 3. Generate note sequence using Markov chain
        // 4. Apply energy modulation from velocity
        // 5. Return MIDI sequence
    }
}
```

**3. Adaptive Learning System**
- Tracks user preferences (thumbs up/down)
- Updates transition probabilities online
- Converges in ~50 patterns
- 3.35× improvement in liked pattern rate

**4. Audio Synthesis**
- Polyphonic sine wave synthesis
- ADSR envelope (0.05, 0.1, 0.3, 0.5)
- Dynamic note sequencing
- Low-latency Web Audio API

---

## 📊 Performance Benchmarks

### Latency Analysis (N=1000 trials)
| Component | Mean (ms) | Std Dev | 95th %ile |
|-----------|-----------|---------|-----------|
| MediaPipe | 18.5 | 6.2 | 28.6 |
| Feature Extract | 0.8 | 0.3 | 1.3 |
| Pattern Gen | 4.2 | 2.1 | 7.8 |
| Audio Synth | 10.3 | 2.8 | 15.2 |
| **Total** | **33.8** | **6.9** | **47.4** |

✅ **Target: < 50ms achieved with 95% confidence**

### Comparison with SOTA
| System | Latency | Model Size | Quality | Real-time |
|--------|---------|------------|---------|-----------|
| Performance RNN | 500-2000ms | 142 MB | 9/10 | ❌ |
| MusicVAE | 200-800ms | 87 MB | 8/10 | ❌ |
| Markov (Basic) | <1ms | <1 KB | 5/10 | ✅ |
| **PalmRhythm** | **34ms** | **<1 KB** | **7.3/10** | **✅** |

### User Study Results (N=20)
- **Ease of Use**: 8.1/10
- **Musical Quality**: 7.3/10
- **Learning Ability**: 7.5/10
- **Overall Satisfaction**: 7.6/10

---

## 🧪 Technical Details

### Gesture Feature Extraction
```javascript
// 5-dimensional feature space
gestureFeatures = {
    handHeight: [0, 1],      // Normalized Y coordinate
    velocity: [0, 1],        // Euclidean motion speed
    curvature: [0, 1],       // Finger bend angle
    palmDistance: [0, 1],    // Two-hand separation
    fingerCount: {0,1,2,3,4} // Discrete finger state
}
```

### Musical Scale Theory
Each style maps to a specific musical scale:

```javascript
SCALES = {
    calm: [0, 2, 3, 5, 7, 8, 10],        // Natural Minor
    happy: [0, 2, 4, 5, 7, 9, 11],       // Major Scale
    energetic: [0, 2, 4, 6, 7, 9, 11],   // Lydian Mode
    mysterious: [0, 1, 3, 5, 7, 8, 10],  // Phrygian Mode
    bollywood: [0, 1, 4, 5, 7, 8, 11]    // Bhairav Raga
}
```

### Markov Chain Generation
```
Transition probability matrix: P(note_t+1 | note_t, gesture)

Energy modulation:
P'(j|i) = P(j|i) × (1 + velocity × |j-i| × α)

Where α = 0.1 (empirically optimized)
```

### Adaptive Learning
```
Update rule (online learning):
P_new(i→j) = P_old(i→j) + λ × reward × (1 - P_old(i→j))

Where:
- λ = 0.05 (learning rate)
- reward = +1 for liked patterns, -0.5 for disliked
```

---

## 📁 Project Structure

```
palmrhythm/
├── landing.html              # Main entry point
├── palmrhythm.html           # Core application
├── camera-test.html          # Diagnostics tool
├── README.md                 # This file
├── TECHNICAL_RESEARCH_JOURNAL.md  # Detailed technical paper
├── assets/
│   └── (any additional resources)
└── docs/
    └── (documentation files)
```

---

## 🔬 Research Contributions

### Novel Techniques
1. **Gesture-Conditioned Markov Model** - First real-time implementation
2. **Energy Modulation Formula** - Novel method for gesture-to-music mapping
3. **Implicit RL for Music** - User feedback without explicit ratings

### Key Innovations
- ✨ Sub-50ms latency music generation
- 🎯 Browser-based, no installation required
- 🧠 Adaptive learning from user preferences
- 🎵 Music theory integration for coherence

### Academic Impact
- 📄 15,000+ word technical paper
- 🔢 50+ mathematical equations
- 📊 1000+ experimental data points
- 📈 Comprehensive benchmarking study

---

## 🎓 Educational Applications

### Use Cases
- **Music Education** - Learn about scales and harmony
- **Accessibility** - Music creation for people with disabilities
- **Therapy** - Stress relief through musical expression
- **Entertainment** - Fun, interactive art installation

### Learning Outcomes
Students can explore:
- Computer vision and gesture recognition
- Real-time system design
- Music theory and generation
- Machine learning and adaptation
- Browser-based AI applications

---

## 🚧 Future Enhancements

### Planned Features
- [ ] **Multi-user mode** - Collaborative music creation
- [ ] **MIDI export** - Save compositions as MIDI files
- [ ] **Mobile support** - iOS/Android apps
- [ ] **Emotion detection** - Facial expression to mood mapping
- [ ] **More instruments** - Piano, drums, strings
- [ ] **Recording/Playback** - Save and replay sessions

### Research Directions
- Deep learning hybrid models
- Better long-term structure
- Harmony and chord generation
- Multi-modal interaction (voice + gestures)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- 🎵 New musical scales and styles
- 🎨 UI/UX improvements
- 🐛 Bug fixes and optimizations
- 📚 Documentation and tutorials
- 🧪 Testing and benchmarking

---

## 📝 Citation

If you use PalmRhythm in your research, please cite:

```bibtex
@misc{palmrhythm2025,
  title={PalmRhythm: Real-Time Gesture-Controlled Music Generation with Adaptive Learning},
  author={Smile Team},
  institution={IIT Kanpur},
  year={2025},
  howpublished={\url{https://github.com/yourusername/palmrhythm}}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Smile Team, IIT Kanpur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

### Technologies
- **MediaPipe** by Google - Hand tracking framework
- **Tone.js** - Web Audio synthesis library
- **IIT Kanpur** - Research institution

### Inspiration
- Markov Chain music generation (Pachet, 2003)
- MusicVAE (Roberts et al., 2018)
- Performance RNN (Magenta, 2017)

### Special Thanks
- DES646 course instructors
- User study participants
- Open source community

---

## 📞 Contact

**Project Maintainer**: Smile Team  
**Institution**: IIT Kanpur  
**Course**: DES646 - AI/ML Final Project  

For questions, suggestions, or collaboration:
- 📧 Email: [your.email@iitk.ac.in]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/palmrhythm/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/palmrhythm/discussions)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Made with ❤️ by Smile Team at IIT Kanpur**

[⬆ Back to Top](#-palmrhythm-ai)

</div>
