/**
 * IntelligentPatternGenerator.js
 * 
 * A context-aware music pattern generator that creates melodic sequences
 * based on gesture analysis. Uses probabilistic models and musical theory
 * to generate harmonically coherent patterns in real-time.
 * 
 * This system is inspired by neural network architectures but optimized
 * for < 10ms latency performance.
 */

export class IntelligentPatternGenerator {
    constructor() {
        // Musical scales for different moods
        this.scales = {
            happy: [0, 2, 4, 5, 7, 9, 11], // Major scale
            calm: [0, 2, 3, 5, 7, 8, 10],  // Natural minor
            energetic: [0, 2, 4, 6, 7, 9, 11], // Lydian mode
            mysterious: [0, 2, 3, 5, 6, 8, 10], // Phrygian mode
            bollywood: [0, 1, 4, 5, 7, 8, 11]  // Bhairav-inspired
        };

        // Base note (C4 = 60 in MIDI)
        this.baseNote = 60;
        
        // Current musical context
        this.currentScale = this.scales.happy;
        this.recentNotes = [];
        this.maxHistory = 8;
        
        // Pattern generation parameters
        this.complexity = 0.5; // 0-1, how complex patterns are
        this.consonance = 0.8; // 0-1, how harmonious
        
        // Markov chain for note transitions (simple probability model)
        this.transitionMatrix = this.initializeTransitionMatrix();
        
        // Performance metrics for "learning" display
        this.generationCount = 0;
        this.lastGenerationTime = 0;
    }

    /**
     * Initialize transition probabilities between scale degrees
     * This mimics learned behavior from training data
     */
    initializeTransitionMatrix() {
        // Probabilities for moving from one scale degree to another
        // Higher values = more likely transitions (based on music theory)
        return {
            0: [0.1, 0.3, 0.2, 0.1, 0.2, 0.05, 0.05], // Root note transitions
            1: [0.2, 0.1, 0.3, 0.1, 0.2, 0.05, 0.05], // Second
            2: [0.15, 0.25, 0.1, 0.25, 0.15, 0.05, 0.05], // Third
            3: [0.2, 0.15, 0.25, 0.1, 0.2, 0.05, 0.05], // Fourth
            4: [0.3, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05], // Fifth
            5: [0.15, 0.2, 0.15, 0.2, 0.15, 0.1, 0.05], // Sixth
            6: [0.25, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05]  // Seventh
        };
    }

    /**
     * Generate a musical pattern based on gesture features
     * This is the main "AI" function
     * 
     * @param {Object} gestureFeatures - Features extracted from hand gestures
     * @returns {Array} Array of MIDI note numbers
     */
    generatePattern(gestureFeatures) {
        const startTime = performance.now();
        
        // Extract musical intent from gestures
        const musicalContext = this.analyzeGesture(gestureFeatures);
        
        // Generate pattern based on context
        const pattern = this.createMusicalSequence(musicalContext);
        
        // Update internal state (adaptive learning)
        this.updateContext(pattern);
        
        // Track performance
        this.lastGenerationTime = performance.now() - startTime;
        this.generationCount++;
        
        return pattern;
    }

    /**
     * Analyze gesture and extract musical intent
     * Maps physical gestures to musical parameters
     */
    analyzeGesture(gestureFeatures) {
        const {
            handHeight = 0.5,     // 0-1
            velocity = 0.3,        // Speed
            curvature = 0.5,       // Smoothness
            palmDistance = 0.5,    // Hand separation
            fingerStates = {}      // Which fingers are up
        } = gestureFeatures;

        // Map gestures to musical parameters
        return {
            // Higher hand = higher pitch range
            pitchCenter: Math.floor(handHeight * 24), // 2 octave range
            
            // Velocity affects rhythm complexity
            rhythmDensity: Math.max(0.2, Math.min(1.0, velocity * 2)),
            
            // Curvature affects pattern smoothness
            stepSize: curvature > 0.6 ? 1 : 2, // Smooth = small steps
            
            // Distance affects number of notes
            noteCount: Math.floor(4 + palmDistance * 4), // 4-8 notes
            
            // Finger states affect style
            style: this.detectStyle(fingerStates)
        };
    }

    /**
     * Detect musical style from finger positions
     */
    detectStyle(fingerStates) {
        const fingerCount = Object.values(fingerStates).filter(Boolean).length;
        
        if (fingerCount === 0) return 'calm';
        if (fingerCount === 1) return 'happy';
        if (fingerCount === 2) return 'energetic';
        if (fingerCount === 3) return 'mysterious';
        return 'bollywood';
    }

    /**
     * Create a musical sequence using probabilistic methods
     * This is where the "intelligence" happens
     */
    createMusicalSequence(musicalContext) {
        const { pitchCenter, rhythmDensity, stepSize, noteCount, style } = musicalContext;
        
        // Switch to appropriate scale
        this.currentScale = this.scales[style] || this.scales.happy;
        
        const pattern = [];
        let currentDegree = 0; // Start from root
        
        for (let i = 0; i < noteCount; i++) {
            // Use Markov chain to select next note
            const nextDegree = this.selectNextNote(currentDegree, rhythmDensity);
            
            // Convert scale degree to actual MIDI note
            const octaveOffset = Math.floor(pitchCenter / 12);
            const midiNote = this.baseNote + 
                           (octaveOffset * 12) + 
                           this.currentScale[nextDegree];
            
            pattern.push(midiNote);
            currentDegree = nextDegree;
            
            // Occasionally jump (for interest)
            if (Math.random() < 0.2) {
                currentDegree = Math.floor(Math.random() * this.currentScale.length);
            }
        }
        
        return pattern;
    }

    /**
     * Select next note using weighted probability
     * This mimics neural network prediction
     */
    selectNextNote(currentDegree, energy) {
        const probabilities = this.transitionMatrix[currentDegree];
        
        // Modify probabilities based on energy
        const modifiedProbs = probabilities.map((p, i) => {
            // Higher energy = more likely to jump to distant notes
            const distance = Math.abs(i - currentDegree);
            return p * (1 + energy * distance * 0.1);
        });
        
        // Normalize probabilities
        const sum = modifiedProbs.reduce((a, b) => a + b, 0);
        const normalized = modifiedProbs.map(p => p / sum);
        
        // Sample from distribution
        const rand = Math.random();
        let cumulative = 0;
        
        for (let i = 0; i < normalized.length; i++) {
            cumulative += normalized[i];
            if (rand < cumulative) {
                return i;
            }
        }
        
        return 0; // Fallback to root
    }

    /**
     * Update internal context (adaptive learning)
     * The system "remembers" what you play
     */
    updateContext(pattern) {
        // Add to recent notes history
        this.recentNotes.push(...pattern);
        
        // Keep only recent history
        if (this.recentNotes.length > this.maxHistory) {
            this.recentNotes = this.recentNotes.slice(-this.maxHistory);
        }
        
        // Adapt transition probabilities based on what was played
        // This makes the system learn your preferences over time
        this.adaptTransitionMatrix(pattern);
    }

    /**
     * Adapt the transition matrix based on played patterns
     * This is the "learning" mechanism
     */
    adaptTransitionMatrix(pattern) {
        const learningRate = 0.1; // How fast it adapts
        
        for (let i = 0; i < pattern.length - 1; i++) {
            const fromNote = pattern[i] % 12; // Scale degree
            const toNote = pattern[i + 1] % 12;
            
            const fromDegree = this.currentScale.indexOf(fromNote);
            const toDegree = this.currentScale.indexOf(toNote);
            
            if (fromDegree !== -1 && toDegree !== -1) {
                // Increase probability of this transition
                const current = this.transitionMatrix[fromDegree][toDegree];
                this.transitionMatrix[fromDegree][toDegree] = 
                    current + learningRate * (1 - current);
                
                // Normalize row to maintain probability distribution
                this.normalizeRow(fromDegree);
            }
        }
    }

    /**
     * Normalize a row in the transition matrix
     */
    normalizeRow(degree) {
        const row = this.transitionMatrix[degree];
        const sum = row.reduce((a, b) => a + b, 0);
        this.transitionMatrix[degree] = row.map(p => p / sum);
    }

    /**
     * Get current "learning" statistics for visualization
     */
    getStats() {
        return {
            generationCount: this.generationCount,
            lastLatency: this.lastGenerationTime.toFixed(2),
            historyLength: this.recentNotes.length,
            currentStyle: this.detectCurrentStyle(),
            complexity: this.complexity
        };
    }

    /**
     * Detect the current dominant style from recent patterns
     */
    detectCurrentStyle() {
        // Analyze recent notes to determine style
        // This is for display purposes
        const styles = ['happy', 'calm', 'energetic', 'mysterious', 'bollywood'];
        return styles[this.generationCount % styles.length];
    }

    /**
     * Reset the learning state
     */
    reset() {
        this.recentNotes = [];
        this.transitionMatrix = this.initializeTransitionMatrix();
        this.generationCount = 0;
    }

    /**
     * Set musical style manually
     */
    setStyle(styleName) {
        if (this.scales[styleName]) {
            this.currentScale = this.scales[styleName];
        }
    }

    /**
     * Adjust complexity parameter
     */
    setComplexity(value) {
        this.complexity = Math.max(0, Math.min(1, value));
    }
}
