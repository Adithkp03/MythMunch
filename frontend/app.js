// Truth Verification System - Main Application Logic

class TruthVerificationApp {
  constructor() {
    this.apiConfig = {
      baseUrl: 'http://localhost:8000',
      bearerToken: 'bc7aba495dfdaa88e718dec0d0fba29a2d45eaecac3268453778d027c7419081'
    };

    this.verdictColors = {
      'TRUE': '#22c55e',
      'FALSE': '#ef4444',
      'CONFLICTING_EVIDENCE': '#f97316',
      'PARTIALLY_TRUE': '#eab308',
      'INSUFFICIENT_INFO': '#6b7280'
    };

    this.sourceColors = {
      'wikipedia': '#3b82f6',
      'pubmed': '#10b981',
      'newsapi': '#f59e0b',
      'google_search': '#8b5cf6',
      'gemini': '#ec4899'
    };

    this.init();
  }

  init() {
    this.bindEvents();
    this.initializeSettings();
    this.bindThemeToggle();
  }
  bindThemeToggle() {
    const toggleBtn = document.getElementById("themeToggle");
    const root = document.documentElement;

    // Load saved theme
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      root.setAttribute("data-color-scheme", savedTheme);
      toggleBtn.textContent = savedTheme === "dark" ? "" : "";
    } else {
      // Default to system preference
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        root.setAttribute("data-color-scheme", "dark");
        toggleBtn.textContent = "";
      } else {
        root.setAttribute("data-color-scheme", "light");
        toggleBtn.textContent = "";
      }
    }

    // Toggle handler
    toggleBtn.addEventListener("click", () => {
      const current = root.getAttribute("data-color-scheme");
      const newTheme = current === "dark" ? "light" : "dark";
      root.setAttribute("data-color-scheme", newTheme);
      localStorage.setItem("theme", newTheme);
      toggleBtn.textContent = newTheme === "dark" ? "" : "";
    });
  }
  bindEvents() {
    // Settings toggle
    const settingsToggle = document.getElementById('settings-toggle');
    const settingsContent = document.getElementById('settings-content');
    
    settingsToggle.addEventListener('click', () => {
      const isExpanded = settingsContent.classList.contains('expanded');
      settingsContent.classList.toggle('expanded');
      settingsToggle.classList.toggle('expanded');
    });

    // Range slider
    const maxSourcesSlider = document.getElementById('max-sources');
    const sourcesValue = document.getElementById('sources-value');
    
    maxSourcesSlider.addEventListener('input', (e) => {
      sourcesValue.textContent = e.target.value;
    });

    // Verify button
    const verifyBtn = document.getElementById('verify-btn');
    verifyBtn.addEventListener('click', () => {
      this.handleVerifyClick();
    });

    // Enter key on textarea
    const claimInput = document.getElementById('claim-input');
    claimInput.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter') {
        this.handleVerifyClick();
      }
    });
  }

  initializeSettings() {
    // Show settings panel by default
    const settingsContent = document.getElementById('settings-content');
    const settingsToggle = document.getElementById('settings-toggle');
    settingsContent.classList.add('expanded');
    settingsToggle.classList.add('expanded');
  }

  validateInput() {
    const claimInput = document.getElementById('claim-input');
    const inputError = document.getElementById('input-error');
    const claim = claimInput.value.trim();

    if (!claim) {
      inputError.classList.remove('hidden');
      claimInput.focus();
      return false;
    }

    inputError.classList.add('hidden');
    return true;
  }

  getFormData() {
    const claim = document.getElementById('claim-input').value.trim();
    const maxSources = parseInt(document.getElementById('max-sources').value);
    const includeEvidence = document.getElementById('include-evidence').checked;
    const advancedAnalysis = document.getElementById('advanced-analysis').checked;

    return {
      claim,
      max_evidence_sources: maxSources,
      include_evidence: includeEvidence,
      detailed_analysis: advancedAnalysis
    };
  }

  async handleVerifyClick() {
    if (!this.validateInput()) {
      return;
    }

    const formData = this.getFormData();
    this.showLoading();

    try {
      const result = await this.makeAPICall(formData);
      this.displayResults(result);
    } catch (error) {
      this.showError(error.message);
    } finally {
      this.hideLoading();
    }
  }

  async makeAPICall(data) {
    const response = await fetch(`${this.apiConfig.baseUrl}/fact-check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiConfig.bearerToken}`
      },
      body: JSON.stringify(data)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  }

  showLoading() {
    const verifyBtn = document.getElementById('verify-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    verifyBtn.classList.add('loading');
    verifyBtn.disabled = true;
    loadingOverlay.classList.remove('hidden');
  }

  hideLoading() {
    const verifyBtn = document.getElementById('verify-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    verifyBtn.classList.remove('loading');
    verifyBtn.disabled = false;
    loadingOverlay.classList.add('hidden');
  }

  displayResults(data) {
    this.displayVerdict(data);
    this.displayEvidence(data);
    this.scrollToResults();
  }

  displayVerdict(data) {
    const resultsSection = document.getElementById('results-section');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const explanationText = document.getElementById('explanation-text');
    const geminiComparison = document.getElementById('gemini-comparison');
    const processingTime = document.getElementById('processing-time');

    // Show results section
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('fade-in');

    // Set verdict
    const verdict = data.verdict || 'INSUFFICIENT_INFO';
    verdictBadge.textContent = verdict.replace('_', ' ');
    verdictBadge.className = `verdict-badge ${this.getVerdictClass(verdict)}`;

    // Set confidence
    const confidence = Math.round((data.confidence || 0) * 100);
    confidenceFill.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;

    // Set explanation
    explanationText.textContent = data.explanation || 'No explanation provided.';

    // Set Gemini comparison if available
    if (data.gemini_verdict) {
      geminiComparison.innerHTML = `
        <strong>Gemini Analysis:</strong> ${data.gemini_verdict}
      `;
      geminiComparison.style.display = 'block';
    } else {
      geminiComparison.style.display = 'none';
    }

    // Set processing time
    if (data.processing_time) {
      processingTime.textContent = `Processed in ${data.processing_time.toFixed(2)}s`;
    }
  }

  displayEvidence(data) {
    const evidenceSection = document.getElementById('evidence-section');
    const evidenceGrid = document.getElementById('evidence-grid');
    const evidenceCount = document.getElementById('evidence-count');

    if (!data.evidence || data.evidence.length === 0) {
      evidenceSection.classList.add('hidden');
      return;
    }

    // Show evidence section
    evidenceSection.classList.remove('hidden');
    evidenceSection.classList.add('fade-in');

    // Set evidence count
    evidenceCount.textContent = `${data.evidence.length} sources found`;

    // Clear existing evidence
    evidenceGrid.innerHTML = '';

    // Create evidence cards
    data.evidence.forEach((evidence, index) => {
      const card = this.createEvidenceCard(evidence, index);
      evidenceGrid.appendChild(card);
    });
  }
  
  createEvidenceCard(evidence, index) {
    
    
    const card = document.createElement('div');
    card.className = 'card evidence-card slide-in';
    card.style.animationDelay = `${index * 0.1}s`;

    const sourceClass = this.getSourceClass(evidence.source_type);
    const credibilityScore = evidence.credibility_score ? Math.round(evidence.credibility_score * 100) : 'N/A';
    const similarityScore = evidence.similarity_score ? Math.round(evidence.similarity_score * 100) : 'N/A';

    card.innerHTML = `
      <div class="card__body">
        <div class="evidence-header-info">
          <span class="evidence-source ${sourceClass}">
            ${evidence.source_type || 'Unknown'}
          </span>
          <div class="evidence-scores">
            <span class="score-badge">Credibility: ${credibilityScore}%</span>
            <span class="score-badge">Similarity: ${similarityScore}%</span>
          </div>
        </div>
        <h4 class="evidence-title">${evidence.title || 'Untitled'}</h4>
        <p class="evidence-snippet truncated">${evidence.snippet || 'No snippet available.'}</p>
        ${evidence.url ? `<a href="${evidence.url}" target="_blank" class="evidence-url">View Source →</a>` : ''}
      </div>
    `;

    // Add click handler for expansion
    card.addEventListener('click', (e) => {
      if (e.target.tagName !== 'A') {
        this.toggleEvidenceCard(card);
      }
    });

    return card;
  }

  toggleEvidenceCard(card) {
    const snippet = card.querySelector('.evidence-snippet');
    const isExpanded = card.classList.contains('expanded');

    if (isExpanded) {
      snippet.classList.add('truncated');
      card.classList.remove('expanded');
    } else {
      snippet.classList.remove('truncated');
      card.classList.add('expanded');
    }
  }

  getVerdictClass(verdict) {
    const classMap = {
      'TRUE': 'verdict-true',
      'FALSE': 'verdict-false',
      'CONFLICTING_EVIDENCE': 'verdict-conflicting',
      'PARTIALLY_TRUE': 'verdict-partially',
      'INSUFFICIENT_INFO': 'verdict-insufficient'
    };
    return classMap[verdict] || 'verdict-insufficient';
  }

  getSourceClass(sourceType) {
    if (!sourceType) return '';
    
    const normalizedSource = sourceType.toLowerCase().replace(/\s+/g, '_');
    return `source-${normalizedSource}`;
  }

  scrollToResults() {
    const resultsSection = document.getElementById('results-section');
    resultsSection.scrollIntoView({ 
      behavior: 'smooth',
      block: 'start'
    });
  }

  showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: var(--color-error);
      color: white;
      padding: var(--space-16) var(--space-20);
      border-radius: var(--radius-base);
      box-shadow: var(--shadow-lg);
      z-index: 1001;
      max-width: 400px;
      animation: slideInRight 0.3s ease-out;
    `;

    errorDiv.innerHTML = `
      <div style="display: flex; align-items: center; gap: var(--space-12);">
        <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
        </svg>
        <div>
          <strong>Error</strong><br>
          ${message}
        </div>
      </div>
    `;

    // Add slide in animation
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideInRight {
        from {
          transform: translateX(100%);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
    `;
    document.head.appendChild(style);

    document.body.appendChild(errorDiv);

    // Auto remove after 5 seconds
    setTimeout(() => {
      errorDiv.style.animation = 'slideInRight 0.3s ease-out reverse';
      setTimeout(() => {
        if (errorDiv.parentNode) {
          errorDiv.parentNode.removeChild(errorDiv);
        }
        if (style.parentNode) {
          style.parentNode.removeChild(style);
        }
      }, 300);
    }, 5000);

    // Click to dismiss
    errorDiv.addEventListener('click', () => {
      errorDiv.style.animation = 'slideInRight 0.3s ease-out reverse';
      setTimeout(() => {
        if (errorDiv.parentNode) {
          errorDiv.parentNode.removeChild(errorDiv);
        }
      }, 300);
    });
  }
}
class CrisisDashboard extends TruthVerificationApp {
    constructor() {
        super();
        this.dashboardMode = false;
        this.monitoringActive = false;
        this.alertsPolling = null;
        this.trendsPolling = null;
    }
    
    init() {
        super.init();
        this.initDashboard();
        this.bindDashboardEvents();
    }
    
    initDashboard() {
        // Add dashboard toggle
        const header = document.querySelector('.app-header .container');
        const dashboardToggle = document.createElement('button');
        dashboardToggle.id = 'dashboard-toggle';
        dashboardToggle.className = 'btn btn--secondary';
        dashboardToggle.textContent = '📊 Crisis Dashboard';
        header.appendChild(dashboardToggle);
        
        // Create dashboard container
        const dashboardHTML = `
            <div id="crisis-dashboard" class="dashboard-container hidden">
                <div class="dashboard-header">
                    <h2>🚨 Crisis Misinformation Monitoring</h2>
                    <div class="dashboard-controls">
                        <button id="start-monitoring" class="btn btn--primary">Start Monitoring</button>
                        <button id="stop-monitoring" class="btn btn--secondary">Stop Monitoring</button>
                        <select id="crisis-selector" class="form-control">
                            <option value="all">All Crises</option>
                            <option value="ukraine_war">Ukraine Conflict</option>
                            <option value="covid_pandemic">COVID-19</option>
                            <option value="climate_crisis">Climate Change</option>
                        </select>
                    </div>
                </div>
                
                <div class="dashboard-grid">
                    <div class="dashboard-card alerts-card">
                        <h3>🚨 Active Alerts</h3>
                        <div id="alerts-container"></div>
                    </div>
                    
                    <div class="dashboard-card trends-card">
                        <h3>📈 Trending Topics</h3>
                        <div id="trends-container"></div>
                    </div>
                    
                    <div class="dashboard-card monitoring-card">
                        <h3>👁️ Live Monitoring</h3>
                        <div id="monitoring-status">Not monitoring</div>
                        <div id="monitoring-stats"></div>
                    </div>
                    
                    <div class="dashboard-card analysis-card">
                        <h3>🔍 Trend Analysis</h3>
                        <div id="analysis-container"></div>
                    </div>
                </div>
            </div>
        `;
        
        const mainContent = document.querySelector('.main-content');
        mainContent.insertAdjacentHTML('afterbegin', dashboardHTML);
    }
    
    bindDashboardEvents() {
        const dashboardToggle = document.getElementById('dashboard-toggle');
        const dashboard = document.getElementById('crisis-dashboard');
        const inputSection = document.querySelector('.input-section');
        
        dashboardToggle.addEventListener('click', () => {
            this.dashboardMode = !this.dashboardMode;
            
            if (this.dashboardMode) {
                dashboard.classList.remove('hidden');
                inputSection.classList.add('hidden');
                dashboardToggle.textContent = '📝 Fact Checker';
                this.startDashboardPolling();
            } else {
                dashboard.classList.add('hidden');
                inputSection.classList.remove('hidden');
                dashboardToggle.textContent = '📊 Crisis Dashboard';
                this.stopDashboardPolling();
            }
        });
        
        // Start/Stop monitoring
        document.getElementById('start-monitoring').addEventListener('click', () => {
            this.startMonitoring();
        });
        
        document.getElementById('stop-monitoring').addEventListener('click', () => {
            this.stopMonitoring();
        });
    }
    
    async startMonitoring() {
        const keywords = ['ukraine', 'covid', 'vaccine', 'climate change', 'election fraud'];
        
        try {
            const response = await fetch(`${this.apiConfig.baseUrl}/start-monitoring`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(keywords)
            });
            
            if (response.ok) {
                this.monitoringActive = true;
                document.getElementById('monitoring-status').textContent = '🟢 Monitoring Active';
                document.getElementById('start-monitoring').disabled = true;
                document.getElementById('stop-monitoring').disabled = false;
            }
        } catch (error) {
            console.error('Failed to start monitoring:', error);
        }
    }
    
    stopMonitoring() {
        this.monitoringActive = false;
        document.getElementById('monitoring-status').textContent = '🔴 Monitoring Stopped';
        document.getElementById('start-monitoring').disabled = false;
        document.getElementById('stop-monitoring').disabled = true;
    }
    
    startDashboardPolling() {
        // Poll alerts every 10 seconds
        this.alertsPolling = setInterval(() => {
            this.updateAlerts();
        }, 10000);
        
        // Poll trends every 30 seconds
        this.trendsPolling = setInterval(() => {
            this.updateTrends();
        }, 30000);
        
        // Initial load
        this.updateAlerts();
        this.updateTrends();
    }
    
    stopDashboardPolling() {
        if (this.alertsPolling) {
            clearInterval(this.alertsPolling);
        }
        if (this.trendsPolling) {
            clearInterval(this.trendsPolling);
        }
    }
    
    async updateAlerts() {
        try {
            const response = await fetch(`${this.apiConfig.baseUrl}/alerts`);
            const data = await response.json();
            
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.forEach(alert => {
                    const alertElement = this.createAlertElement(alert);
                    container.appendChild(alertElement);
                });
            } else {
                container.innerHTML = '<p class="no-alerts">No active alerts</p>';
            }
        } catch (error) {
            console.error('Failed to update alerts:', error);
        }
    }
    
    async updateTrends() {
        try {
            const response = await fetch(`${this.apiConfig.baseUrl}/trends`);
            const data = await response.json();
            
            const container = document.getElementById('trends-container');
            container.innerHTML = '';
            
            if (data.trending_keywords && data.trending_keywords.length > 0) {
                data.trending_keywords.forEach(([keyword, count]) => {
                    const trendElement = this.createTrendElement(keyword, count);
                    container.appendChild(trendElement);
                });
            } else {
                container.innerHTML = '<p class="no-trends">No trending topics</p>';
            }
        } catch (error) {
            console.error('Failed to update trends:', error);
        }
    }
    
    createAlertElement(alert) {
        const div = document.createElement('div');
        div.className = 'alert-item high-risk';
        div.innerHTML = `
            <div class="alert-header">
                <span class="alert-keyword">${alert.keyword}</span>
                <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="alert-content">${alert.mention.content.substring(0, 100)}...</div>
            <div class="alert-actions">
                <button class="btn btn--sm" onclick="app.investigateAlert('${alert.alert_id}')">
                    🔍 Investigate
                </button>
            </div>
        `;
        return div;
    }
    
    createTrendElement(keyword, count) {
        const div = document.createElement('div');
        div.className = 'trend-item';
        div.innerHTML = `
            <div class="trend-keyword">${keyword}</div>
            <div class="trend-count">${count} mentions</div>
            <div class="trend-bar">
                <div class="trend-fill" style="width: ${Math.min(count / 10 * 100, 100)}%"></div>
            </div>
        `;
        return div;
    }
    
    async investigateAlert(alertId) {
        // This would open a detailed investigation view
        console.log('Investigating alert:', alertId);
        // You can implement a modal or new page for detailed analysis
    }
}

// Replace the original app with the enhanced version
const app = new CrisisDashboard();
// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new TruthVerificationApp();
});

// Add some sample data for testing if needed
window.sampleTestData = {
  "analysis_id": "test-123",
  "claim": "The Earth is flat",
  "verdict": "REFUTES",
  "confidence": 0.95,
  "explanation": "Scientific evidence overwhelmingly supports that the Earth is a sphere. Multiple lines of evidence including satellite imagery, physics of gravity, and direct observation contradict the flat Earth hypothesis.",
  "evidence": [
    {
      "source_type": "wikipedia",
      "title": "Figure of the Earth - Wikipedia",
      "snippet": "The figure of the Earth is the size and shape used to model the Earth. The kind of figure depends on application, including the precision needed for the model...",
      "url": "https://en.wikipedia.org/wiki/Figure_of_the_Earth",
      "credibility_score": 0.9,
      "similarity_score": 0.85
    },
    {
      "source_type": "pubmed",
      "title": "Geodetic evidence for Earth's spherical shape",
      "snippet": "Modern geodetic measurements using satellite technology provide conclusive evidence that Earth approximates an oblate spheroid...",
      "url": "https://pubmed.ncbi.nlm.nih.gov/example",
      "credibility_score": 0.95,
      "similarity_score": 0.92
    }
  ],
  "gemini_verdict": "Scientific consensus strongly refutes flat Earth claims through multiple independent lines of evidence.",
  "processing_time": 2.34
};