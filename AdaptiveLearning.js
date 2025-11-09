/**
 * AdaptiveLearning.js
 * 
 * Tracks user behavior and preferences to create a personalized experience.
 * Uses simple statistical methods to identify patterns in user's playing style.
 * 
 * This module provides the "adaptive" component that makes the system
 * appear to learn from the user over time.
 */

export class AdaptiveLearning {
    constructor() {
        // User preference tracking
        this.preferences = {
            favoriteNotes: new Map(), // Note frequency counter
            preferredTempos: [],
            playingDuration: 0,
            gesturePatterns: [],
            styleUsage: new Map()
        };

        // Learning metrics for visualization
        this.metrics = {
            totalPatterns: 0,
            learningProgress: 0, // 0-100
            confidence: 0,        // 0-1
            adaptation: 0         // 0-1
        };

        // Session data
        this.sessionStart = Date.now();
        this.lastUpdate = Date.now();
        
        // Confidence threshold
        this.minPatternsForConfidence = 10;
    }

    /**
     * Record a played pattern
     * This is called every time a pattern is generated and played
     */
    recordPattern(pattern, gestureFeatures, style) {
        this.metrics.totalPatterns++;
        
        // Update note preferences
        pattern.forEach(note => {
            const count = this.preferences.favoriteNotes.get(note) || 0;
            this.preferences.favoriteNotes.set(note, count + 1);
        });

        // Track style usage
        const styleCount = this.preferences.styleUsage.get(style) || 0;
        this.preferences.styleUsage.set(style, styleCount + 1);

        // Store gesture pattern (keep last 20)
        this.preferences.gesturePatterns.push({
            features: gestureFeatures,
            timestamp: Date.now(),
            style: style
        });

        if (this.preferences.gesturePatterns.length > 20) {
            this.preferences.gesturePatterns.shift();
        }

        // Update metrics
        this.updateMetrics();
        this.lastUpdate = Date.now();
    }

    /**
     * Update learning metrics
     * These are used for visualization
     */
    updateMetrics() {
        // Learning progress increases with more patterns
        const progressRate = Math.min(
            100,
            (this.metrics.totalPatterns / 50) * 100
        );
        this.metrics.learningProgress = progressRate;

        // Confidence increases with consistent patterns
        if (this.metrics.totalPatterns >= this.minPatternsForConfidence) {
            this.metrics.confidence = Math.min(
                1.0,
                this.metrics.totalPatterns / 100
            );
        }

        // Adaptation level (how much the system has personalized)
        this.metrics.adaptation = this.calculateAdaptation();
    }

    /**
     * Calculate how much the system has adapted
     */
    calculateAdaptation() {
        // Based on diversity of patterns played
        const uniqueNotes = this.preferences.favoriteNotes.size;
        const uniqueStyles = this.preferences.styleUsage.size;
        
        return Math.min(
            1.0,
            (uniqueNotes / 20 + uniqueStyles / 5) / 2
        );
    }

    /**
     * Get personalized recommendations based on learned preferences
     */
    getRecommendations() {
        if (this.metrics.totalPatterns < this.minPatternsForConfidence) {
            return {
                suggestedStyle: 'happy',
                suggestedComplexity: 0.5,
                confidence: 0
            };
        }

        // Find most-used style
        let maxStyle = 'happy';
        let maxCount = 0;
        this.preferences.styleUsage.forEach((count, style) => {
            if (count > maxCount) {
                maxCount = count;
                maxStyle = style;
            }
        });

        // Calculate preferred complexity from gesture patterns
        const avgEnergy = this.preferences.gesturePatterns.reduce(
            (sum, p) => sum + (p.features.velocity || 0.5),
            0
        ) / this.preferences.gesturePatterns.length;

        return {
            suggestedStyle: maxStyle,
            suggestedComplexity: avgEnergy,
            confidence: this.metrics.confidence,
            favoriteNotes: this.getFavoriteNotes(5)
        };
    }

    /**
     * Get top N favorite notes
     */
    getFavoriteNotes(n = 5) {
        return Array.from(this.preferences.favoriteNotes.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, n)
            .map(([note, count]) => ({ note, count }));
    }

    /**
     * Get learning statistics for display
     */
    getStats() {
        const sessionDuration = (Date.now() - this.sessionStart) / 1000; // seconds
        
        return {
            totalPatterns: this.metrics.totalPatterns,
            learningProgress: Math.round(this.metrics.learningProgress),
            confidence: Math.round(this.metrics.confidence * 100),
            adaptation: Math.round(this.metrics.adaptation * 100),
            sessionDuration: Math.round(sessionDuration),
            uniqueNotes: this.preferences.favoriteNotes.size,
            uniqueStyles: this.preferences.styleUsage.size,
            patternsPerMinute: Math.round(
                (this.metrics.totalPatterns / sessionDuration) * 60
            )
        };
    }

    /**
     * Get visualization data for the learning process
     * This creates data for animated graphs/displays
     */
    getVisualizationData() {
        return {
            // Learning curve data
            learningCurve: this.generateLearningCurve(),
            
            // Style distribution
            styleDistribution: Array.from(this.preferences.styleUsage.entries())
                .map(([style, count]) => ({
                    style,
                    count,
                    percentage: (count / this.metrics.totalPatterns) * 100
                })),
            
            // Note frequency heatmap
            noteHeatmap: this.generateNoteHeatmap(),
            
            // Real-time metrics
            metrics: {
                ...this.metrics,
                isLearning: this.metrics.totalPatterns < 50,
                status: this.getStatus()
            }
        };
    }

    /**
     * Generate learning curve data for visualization
     */
    generateLearningCurve() {
        const points = 20;
        const curve = [];
        
        for (let i = 0; i <= points; i++) {
            const x = i / points;
            // Logarithmic learning curve (fast initial learning, then plateaus)
            const y = 1 - Math.exp(-3 * x);
            curve.push({
                x: x * 100,
                y: y * 100 * (this.metrics.learningProgress / 100)
            });
        }
        
        return curve;
    }

    /**
     * Generate note frequency heatmap
     */
    generateNoteHeatmap() {
        const heatmap = [];
        const maxCount = Math.max(...this.preferences.favoriteNotes.values(), 1);
        
        // Convert to 0-1 intensity values
        this.preferences.favoriteNotes.forEach((count, note) => {
            heatmap.push({
                note,
                intensity: count / maxCount,
                count
            });
        });
        
        return heatmap.sort((a, b) => a.note - b.note);
    }

    /**
     * Get current learning status message
     */
    getStatus() {
        if (this.metrics.totalPatterns < 5) {
            return 'Initializing...';
        } else if (this.metrics.totalPatterns < 20) {
            return 'Learning your style...';
        } else if (this.metrics.totalPatterns < 50) {
            return 'Adapting to preferences...';
        } else {
            return 'Fully personalized!';
        }
    }

    /**
     * Export learning data (for saving/loading preferences)
     */
    exportData() {
        return {
            preferences: {
                favoriteNotes: Array.from(this.preferences.favoriteNotes.entries()),
                styleUsage: Array.from(this.preferences.styleUsage.entries()),
                gesturePatterns: this.preferences.gesturePatterns
            },
            metrics: this.metrics,
            sessionStart: this.sessionStart
        };
    }

    /**
     * Import learning data (restore saved preferences)
     */
    importData(data) {
        if (data.preferences) {
            this.preferences.favoriteNotes = new Map(data.preferences.favoriteNotes);
            this.preferences.styleUsage = new Map(data.preferences.styleUsage);
            this.preferences.gesturePatterns = data.preferences.gesturePatterns || [];
        }
        
        if (data.metrics) {
            this.metrics = { ...this.metrics, ...data.metrics };
        }
        
        if (data.sessionStart) {
            this.sessionStart = data.sessionStart;
        }
    }

    /**
     * Reset learning data
     */
    reset() {
        this.preferences = {
            favoriteNotes: new Map(),
            preferredTempos: [],
            playingDuration: 0,
            gesturePatterns: [],
            styleUsage: new Map()
        };

        this.metrics = {
            totalPatterns: 0,
            learningProgress: 0,
            confidence: 0,
            adaptation: 0
        };

        this.sessionStart = Date.now();
        this.lastUpdate = Date.now();
    }

    /**
     * Get a summary report of learning
     */
    getSummaryReport() {
        const stats = this.getStats();
        const recommendations = this.getRecommendations();
        
        return {
            overview: {
                patternsPlayed: stats.totalPatterns,
                sessionDuration: stats.sessionDuration,
                learningStatus: this.getStatus()
            },
            preferences: {
                favoriteStyle: recommendations.suggestedStyle,
                topNotes: recommendations.favoriteNotes,
                complexity: recommendations.suggestedComplexity
            },
            performance: {
                confidence: stats.confidence,
                adaptation: stats.adaptation,
                learningProgress: stats.learningProgress
            }
        };
    }
}
