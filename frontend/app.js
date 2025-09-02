/**
 * MythMunch AI - Clean Professional Version
 * Enhanced Truth Verification System (No Emojis)
 * Version: 2.1.1
 */

class CrisisGuardApp {
  constructor() {
    this.apiConfig = {
      baseUrl: 'http://localhost:8000',
      bearerToken: 'bc7aba495dfdaa88e718dec0d0fba29a2d45eaecac3268453778d027c7419081',
      endpoints: {
        factCheck: '/fact-check',
        crisisFactCheck: '/crisis-fact-check',
        alerts: '/alerts',
        trends: '/trends',
        health: '/health',
        startMonitoring: '/start-monitoring',
        stopMonitoring: '/stop-monitoring',
        analyzeTrends: '/analyze-trends'
      }
    };

    this.state = {
      currentTab: 'fact-check',
      isMonitoring: false,
      connectionStatus: 'connected',
      alertsData: [],
      trendsData: [],
      chartInstances: {},
      activityFeed: [],
      lastUpdate: null
    };

    this.verdictColors = {
      'TRUE': { bg: '#10b981', text: 'white' },
      'FALSE': { bg: '#ef4444', text: 'white' },
      'CONFLICTING_EVIDENCE': { bg: '#f59e0b', text: 'white' },
      'PARTIALLY_TRUE': { bg: '#eab308', text: 'white' },
      'INSUFFICIENT_INFO': { bg: '#6b7280', text: 'white' }
    };

    this.sourceColors = {
      'wikipedia': '#3b82f6',
      'pubmed': '#10b981',
      'newsapi': '#f59e0b',
      'google_search': '#8b5cf6',
      'gemini': '#ec4899',
      'twitter': '#1da1f2',
      'reddit': '#ff4500'
    };

    this.quickExamples = {
      'covid-vaccine': 'COVID-19 vaccines contain microchips for tracking people',
      'ukraine-war': 'Ukraine war footage is completely staged and fake',
      'climate-change': 'Climate change is a hoax created by scientists for funding',
      'election-fraud': 'The 2024 election results were manipulated by voting machines'
    };

    this.activityIcons = {
      'system': 'settings',
      'analysis': 'search',
      'monitoring': 'activity',
      'input': 'edit-3',
      'navigation': 'compass',
      'error': 'x-circle',
      'emergency': 'alert-triangle'
    };

    this.init();
  }

  /**
   * Initialize the application
   */
  async init() {
    console.log('Initializing MythMunch AI...');
    
    try {
      await this.setupEventListeners();
      await this.initializeCharts();
      await this.loadSavedSettings();
      await this.startPeriodicUpdates();
      await this.checkSystemHealth();
      
      console.log('MythMunch AI initialized successfully');
      this.addActivityFeedItem('system', 'System initialized successfully');
    } catch (error) {
      console.error('Failed to initialize:', error);
      this.showErrorNotification('Failed to initialize system');
    }
  }

  /**
   * Setup all event listeners
   */
  async setupEventListeners() {
    // Navigation tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        const tabId = e.currentTarget.dataset.tab;
        this.switchTab(tabId);
      });
    });

    // Theme toggle
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => this.toggleTheme());
    }

    // Emergency alert
    const emergencyBtn = document.getElementById('emergency-alert');
    if (emergencyBtn) {
      emergencyBtn.addEventListener('click', () => this.showEmergencyModal());
    }

    // Quick action buttons
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const example = e.currentTarget.dataset.example;
        this.loadExample(example);
      });
    });

    // Voice input
    const voiceBtn = document.getElementById('voice-input-btn');
    if (voiceBtn) {
      voiceBtn.addEventListener('click', () => this.startVoiceInput());
    }

    // Paste button
    const pasteBtn = document.getElementById('paste-btn');
    if (pasteBtn) {
      pasteBtn.addEventListener('click', () => this.pasteFromClipboard());
    }

    // Character counter
    const claimInput = document.getElementById('claim-input');
    if (claimInput) {
      claimInput.addEventListener('input', (e) => this.updateCharCounter(e.target.value));
      claimInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
          this.handleVerifyClick();
        }
      });
    }

    // Clear button
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearInput());
    }

    // Sentiment analysis
    const sentimentBtn = document.getElementById('analyze-sentiment');
    if (sentimentBtn) {
      sentimentBtn.addEventListener('click', () => this.analyzeSentiment());
    }

    // Settings panel toggle
    const settingsToggle = document.getElementById('settings-toggle');
    if (settingsToggle) {
      settingsToggle.addEventListener('click', () => this.toggleSettings());
    }

    // Range slider
    const maxSourcesSlider = document.getElementById('max-sources');
    if (maxSourcesSlider) {
      maxSourcesSlider.addEventListener('input', (e) => {
        document.getElementById('sources-value').textContent = e.target.value;
      });
    }

    // Verify button
    const verifyBtn = document.getElementById('verify-btn');
    if (verifyBtn) {
      verifyBtn.addEventListener('click', () => this.handleVerifyClick());
    }

    // Save draft
    const saveDraftBtn = document.getElementById('save-draft-btn');
    if (saveDraftBtn) {
      saveDraftBtn.addEventListener('click', () => this.saveDraft());
    }

    // Share and export results
    const shareBtn = document.getElementById('share-results');
    if (shareBtn) {
      shareBtn.addEventListener('click', () => this.shareResults());
    }

    const exportBtn = document.getElementById('export-results');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportResults());
    }

    // Dashboard controls
    const startMonitoringBtn = document.getElementById('start-monitoring');
    if (startMonitoringBtn) {
      startMonitoringBtn.addEventListener('click', () => this.startMonitoring());
    }

    const stopMonitoringBtn = document.getElementById('stop-monitoring');
    if (stopMonitoringBtn) {
      stopMonitoringBtn.addEventListener('click', () => this.stopMonitoring());
    }

    // Crisis filter
    const crisisFilter = document.getElementById('crisis-filter');
    if (crisisFilter) {
      crisisFilter.addEventListener('change', (e) => this.filterCrisis(e.target.value));
    }

    // Trend timeframe
    const trendTimeframe = document.getElementById('trend-timeframe');
    if (trendTimeframe) {
      trendTimeframe.addEventListener('change', (e) => this.updateTrendTimeframe(e.target.value));
    }

    // Evidence filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.filterEvidence(e.target.dataset.filter);
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
      });
    });

    // Chart controls
    document.querySelectorAll('.chart-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.switchChartMetric(e.target.dataset.metric);
        document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
      });
    });

    // Modal close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.target.closest('.modal-overlay').classList.add('hidden');
      });
    });

    // Refresh buttons
    document.querySelectorAll('.refresh-btn').forEach(btn => {
      btn.addEventListener('click', () => this.refreshDashboard());
    });

    // Pause feed
    const pauseFeedBtn = document.getElementById('pause-feed');
    if (pauseFeedBtn) {
      pauseFeedBtn.addEventListener('click', (e) => this.toggleActivityFeed(e.target));
    }
  }

  /**
   * Initialize charts
   */
  async initializeCharts() {
    try {
      // Real-time chart
      const realtimeCtx = document.getElementById('realtime-chart');
      if (realtimeCtx) {
        this.state.chartInstances.realtime = new Chart(realtimeCtx, {
          type: 'line',
          data: {
            labels: [],
            datasets: [{
              label: 'Mentions',
              data: [],
              borderColor: '#00d4ff',
              backgroundColor: 'rgba(0, 212, 255, 0.1)',
              tension: 0.4,
              fill: true
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: false
              }
            },
            scales: {
              x: {
                display: true,
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              },
              y: {
                display: true,
                beginAtZero: true,
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              }
            },
            elements: {
              point: {
                radius: 0,
                hitRadius: 10,
                hoverRadius: 5
              }
            }
          }
        });
      }

      // Accuracy chart
      const accuracyCtx = document.getElementById('accuracy-chart');
      if (accuracyCtx) {
        this.state.chartInstances.accuracy = new Chart(accuracyCtx, {
          type: 'doughnut',
          data: {
            labels: ['True', 'False', 'Insufficient Info', 'Partially True'],
            datasets: [{
              data: [45, 25, 20, 10],
              backgroundColor: ['#10b981', '#ef4444', '#6b7280', '#eab308'],
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  usePointStyle: true,
                  padding: 20
                }
              }
            }
          }
        });
      }

      // Sources chart
      const sourcesCtx = document.getElementById('sources-chart');
      if (sourcesCtx) {
        this.state.chartInstances.sources = new Chart(sourcesCtx, {
          type: 'bar',
          data: {
            labels: ['Wikipedia', 'PubMed', 'News API', 'Google', 'Social Media'],
            datasets: [{
              label: 'Reliability Score',
              data: [95, 88, 72, 85, 45],
              backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'],
              borderRadius: 4
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                display: false
              }
            },
            scales: {
              y: {
                beginAtZero: true,
                max: 100,
                grid: {
                  color: 'rgba(255, 255, 255, 0.1)'
                }
              },
              x: {
                grid: {
                  display: false
                }
              }
            }
          }
        });
      }

      console.log('Charts initialized successfully');
    } catch (error) {
      console.error('Failed to initialize charts:', error);
    }
  }

  /**
   * Load saved settings from localStorage
   */
  async loadSavedSettings() {
    try {
      const savedTheme = localStorage.getItem('crisisguard-theme');
      if (savedTheme) {
        document.documentElement.setAttribute('data-color-scheme', savedTheme);
        this.updateThemeToggle(savedTheme);
      }

      const savedDrafts = localStorage.getItem('crisisguard-drafts');
      if (savedDrafts) {
        this.drafts = JSON.parse(savedDrafts);
      }

      const savedSettings = localStorage.getItem('crisisguard-settings');
      if (savedSettings) {
        const settings = JSON.parse(savedSettings);
        this.applySettings(settings);
      }

      console.log('Settings loaded successfully');
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  }

  /**
   * Start periodic updates
   */
  async startPeriodicUpdates() {
    // Update stats every 30 seconds
    setInterval(() => {
      if (this.state.currentTab === 'dashboard') {
        this.updateDashboardStats();
        this.updateRealtimeChart();
      }
    }, 30000);

    // Update alerts every 15 seconds
    setInterval(() => {
      if (this.state.isMonitoring) {
        this.updateAlerts();
        this.updateTrends();
      }
    }, 15000);

    // Update connection status every 5 seconds
    setInterval(() => {
      this.checkConnectionStatus();
    }, 5000);

    console.log('Periodic updates started');
  }

  /**
   * Check system health
   */
  async checkSystemHealth() {
    try {
      const response = await this.makeAPICall('/health');
      if (response) {
        this.updateConnectionStatus('connected');
        this.updateHealthMetrics(response);
      }
    } catch (error) {
      this.updateConnectionStatus('disconnected');
      console.error('Health check failed:', error);
    }
  }

  /**
   * Switch between tabs
   */
  switchTab(tabId) {
    // Update navigation
    document.querySelectorAll('.nav-tab').forEach(tab => {
      tab.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.remove('active');
    });
    document.getElementById(`${tabId}-tab`).classList.add('active');

    this.state.currentTab = tabId;

    // Load tab-specific data
    if (tabId === 'dashboard') {
      this.refreshDashboard();
    } else if (tabId === 'analytics') {
      this.loadAnalytics();
    }

    this.addActivityFeedItem('navigation', `Switched to ${tabId} tab`);
  }

  /**
   * Toggle theme
   */
  toggleTheme() {
    const root = document.documentElement;
    const current = root.getAttribute('data-color-scheme');
    const newTheme = current === 'dark' ? 'light' : 'dark';
    
    root.setAttribute('data-color-scheme', newTheme);
    localStorage.setItem('crisisguard-theme', newTheme);
    this.updateThemeToggle(newTheme);
    
    this.addActivityFeedItem('system', `Switched to ${newTheme} theme`);
  }

  /**
   * Update theme toggle button
   */
  updateThemeToggle(theme) {
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
      const svg = toggle.querySelector('svg path');
      if (theme === 'dark') {
        // Sun icon for light mode activation
        toggle.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="5"/>
            <line x1="12" y1="1" x2="12" y2="3"/>
            <line x1="12" y1="21" x2="12" y2="23"/>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
            <line x1="1" y1="12" x2="3" y2="12"/>
            <line x1="21" y1="12" x2="23" y2="12"/>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
          </svg>
        `;
      } else {
        // Moon icon for dark mode activation
        toggle.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
          </svg>
        `;
      }
    }
  }

  /**
   * Show emergency modal
   */
  showEmergencyModal() {
    const modal = document.getElementById('emergency-modal');
    if (modal) {
      modal.classList.remove('hidden');
      this.addActivityFeedItem('emergency', 'Emergency alert system activated');
    }
  }

  /**
   * Load example claim
   */
  loadExample(example) {
    const claimInput = document.getElementById('claim-input');
    if (claimInput && this.quickExamples[example]) {
      claimInput.value = this.quickExamples[example];
      this.updateCharCounter(claimInput.value);
      
      // Scroll to form
      claimInput.scrollIntoView({ behavior: 'smooth' });
      claimInput.focus();
      
      this.addActivityFeedItem('input', `Loaded ${example} example`);
    }
  }

  /**
   * Start voice input (Web Speech API)
   */
  startVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      this.showErrorNotification('Voice input not supported in this browser');
      return;
    }

    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    const voiceBtn = document.getElementById('voice-input-btn');
    const originalHTML = voiceBtn.innerHTML;
    voiceBtn.innerHTML = `
      <svg class="btn-icon animate-pulse" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="m12 8-9.04 9.06a2.82 2.82 0 1 0 3.98 3.98L16 12"/>
        <circle cx="12" cy="8" r="2"/>
      </svg>
      Listening...
    `;
    voiceBtn.disabled = true;

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      const claimInput = document.getElementById('claim-input');
      if (claimInput) {
        claimInput.value = transcript;
        this.updateCharCounter(transcript);
        this.addActivityFeedItem('input', 'Voice input captured');
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      this.showErrorNotification('Voice input failed: ' + event.error);
    };

    recognition.onend = () => {
      voiceBtn.innerHTML = originalHTML;
      voiceBtn.disabled = false;
    };

    recognition.start();
  }

  /**
   * Paste from clipboard
   */
  async pasteFromClipboard() {
    try {
      const text = await navigator.clipboard.readText();
      const claimInput = document.getElementById('claim-input');
      if (claimInput && text.trim()) {
        claimInput.value = text.trim();
        this.updateCharCounter(text.trim());
        this.addActivityFeedItem('input', 'Content pasted from clipboard');
      }
    } catch (error) {
      console.error('Failed to paste:', error);
      this.showErrorNotification('Failed to paste from clipboard');
    }
  }

  /**
   * Update character counter
   */
  updateCharCounter(text) {
    const counter = document.getElementById('char-count');
    if (counter) {
      counter.textContent = text.length;
      
      // Update color based on length
      if (text.length > 1800) {
        counter.style.color = '#ef4444';
      } else if (text.length > 1500) {
        counter.style.color = '#f59e0b';
      } else {
        counter.style.color = 'var(--color-text-tertiary)';
      }
    }
  }

  /**
   * Clear input
   */
  clearInput() {
    const claimInput = document.getElementById('claim-input');
    if (claimInput) {
      claimInput.value = '';
      this.updateCharCounter('');
      claimInput.focus();
      this.addActivityFeedItem('input', 'Input cleared');
    }
  }

  /**
   * Analyze sentiment of the claim
   */
  async analyzeSentiment() {
    const claimInput = document.getElementById('claim-input');
    const sentimentBtn = document.getElementById('analyze-sentiment');
    
    if (!claimInput || !claimInput.value.trim()) {
      this.showErrorNotification('Please enter a claim first');
      return;
    }

    try {
      sentimentBtn.disabled = true;
      
      // Simple sentiment analysis (in a real app, this would call an API)
      const text = claimInput.value.toLowerCase();
      const positiveWords = ['good', 'great', 'excellent', 'positive', 'beneficial', 'helpful'];
      const negativeWords = ['bad', 'terrible', 'awful', 'negative', 'harmful', 'dangerous', 'fake', 'false'];
      
      let sentiment = 'neutral';
      let score = 0;
      
      positiveWords.forEach(word => {
        if (text.includes(word)) score += 1;
      });
      
      negativeWords.forEach(word => {
        if (text.includes(word)) score -= 1;
      });
      
      if (score > 0) sentiment = 'positive';
      else if (score < 0) sentiment = 'negative';
      
      // Update button based on sentiment
      const sentimentIcon = sentimentBtn.querySelector('svg');
      if (sentimentIcon) {
        if (sentiment === 'positive') {
          sentimentIcon.innerHTML = `
            <circle cx="12" cy="12" r="10"/>
            <path d="m8 14s1.5 2 4 2 4-2 4-2"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
          `;
        } else if (sentiment === 'negative') {
          sentimentIcon.innerHTML = `
            <circle cx="12" cy="12" r="10"/>
            <path d="m16 16s-1.5-2-4-2-4 2-4 2"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
          `;
        } else {
          sentimentIcon.innerHTML = `
            <circle cx="12" cy="12" r="10"/>
            <line x1="8" y1="15" x2="16" y2="15"/>
            <line x1="9" y1="9" x2="9.01" y2="9"/>
            <line x1="15" y1="9" x2="15.01" y2="9"/>
          `;
        }
      }
      
      this.addActivityFeedItem('analysis', `Sentiment analyzed: ${sentiment}`);
    } catch (error) {
      console.error('Sentiment analysis failed:', error);
    } finally {
      sentimentBtn.disabled = false;
    }
  }

  /**
   * Toggle settings panel
   */
  toggleSettings() {
    const content = document.getElementById('settings-content');
    const toggle = document.getElementById('settings-toggle');
    
    if (content && toggle) {
      const isExpanded = content.classList.contains('expanded');
      content.classList.toggle('expanded');
      toggle.classList.toggle('expanded');
      
      const arrow = toggle.querySelector('.settings-arrow');
      if (arrow) {
        if (isExpanded) {
          arrow.style.transform = 'rotate(-90deg)';
        } else {
          arrow.style.transform = 'rotate(0deg)';
        }
      }
    }
  }

  /**
   * Handle verify click
   */
  async handleVerifyClick() {
    const claimInput = document.getElementById('claim-input');
    
    if (!this.validateInput()) {
      return;
    }

    const formData = this.getFormData();
    
    try {
      this.showLoading();
      this.addActivityFeedItem('analysis', 'Starting fact-check analysis');
      
      const result = await this.makeFactCheckCall(formData);
      
      if (result) {
        this.displayResults(result);
        this.addActivityFeedItem('analysis', 'Fact-check completed successfully');
        this.updateStats();
      }
    } catch (error) {
      console.error('Fact-check failed:', error);
      this.showError(error.message);
      this.addActivityFeedItem('error', `Fact-check failed: ${error.message}`);
    } finally {
      this.hideLoading();
    }
  }

  /**
   * Validate input
   */
  validateInput() {
    const claimInput = document.getElementById('claim-input');
    const inputError = document.getElementById('input-error');
    
    if (!claimInput || !claimInput.value.trim()) {
      if (inputError) {
        inputError.classList.remove('hidden');
      }
      if (claimInput) {
        claimInput.focus();
      }
      return false;
    }

    if (claimInput.value.length < 10) {
      this.showErrorNotification('Claim must be at least 10 characters long');
      return false;
    }

    if (inputError) {
      inputError.classList.add('hidden');
    }
    return true;
  }

  /**
   * Get form data
   */
  getFormData() {
    const claim = document.getElementById('claim-input')?.value.trim() || '';
    const maxSources = parseInt(document.getElementById('max-sources')?.value || '12');
    const includeEvidence = document.getElementById('include-evidence')?.checked || true;
    const socialMonitoring = document.getElementById('social-monitoring')?.checked || true;
    const realtimeCheck = document.getElementById('real-time-check')?.checked || false;
    const biasDetection = document.getElementById('bias-detection')?.checked || true;
    const analysisMode = document.getElementById('analysis-mode')?.value || 'standard';

    return {
      claim,
      max_evidence_sources: maxSources,
      include_evidence: includeEvidence,
      social_media_monitoring: socialMonitoring,
      real_time_cross_checking: realtimeCheck,
      bias_detection: biasDetection,
      analysis_mode: analysisMode,
      detailed_analysis: analysisMode !== 'standard'
    };
  }

  /**
   * Make fact-check API call
   */
  async makeFactCheckCall(data) {
    const endpoint = data.detailed_analysis ? 
      this.apiConfig.endpoints.crisisFactCheck : 
      this.apiConfig.endpoints.factCheck;

    return await this.makeAPICall(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.apiConfig.bearerToken && {
          'Authorization': `Bearer ${this.apiConfig.bearerToken}`
        })
      },
      body: JSON.stringify(data)
    });
  }

  /**
   * Generic API call method
   */
  async makeAPICall(endpoint, options = {}) {
    try {
      const url = `${this.apiConfig.baseUrl}${endpoint}`;
      const defaultOptions = {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      };

      const finalOptions = { ...defaultOptions, ...options };
      
      console.log(`API Call: ${finalOptions.method} ${url}`);
      
      const response = await fetch(url, finalOptions);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      console.log(`API Response: ${endpoint}`, result);
      
      return result;
    } catch (error) {
      console.error(`API Error: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Show loading overlay
   */
  showLoading() {
    const overlay = document.getElementById('loading-overlay');
    const verifyBtn = document.getElementById('verify-btn');
    const progressBar = document.getElementById('loading-progress');
    const progressText = document.getElementById('progress-text');

    if (overlay) {
      overlay.classList.remove('hidden');
    }

    if (verifyBtn) {
      verifyBtn.classList.add('loading');
      verifyBtn.disabled = true;
    }

    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 100) progress = 100;

      if (progressBar) {
        progressBar.style.width = `${progress}%`;
      }
      if (progressText) {
        progressText.textContent = `${Math.round(progress)}%`;
      }

      if (progress >= 100) {
        clearInterval(progressInterval);
      }
    }, 200);

    this.progressInterval = progressInterval;
  }

  /**
   * Hide loading overlay
   */
  hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    const verifyBtn = document.getElementById('verify-btn');

    if (this.progressInterval) {
      clearInterval(this.progressInterval);
    }

    if (overlay) {
      overlay.classList.add('hidden');
    }

    if (verifyBtn) {
      verifyBtn.classList.remove('loading');
      verifyBtn.disabled = false;
    }
  }

  /**
   * Display verification results
   */
  displayResults(data) {
    this.displayVerdict(data);
    this.displayEvidence(data);
    this.displayInterventions(data);
    this.scrollToResults();
  }

  /**
   * Display verdict information
   */
  displayVerdict(data) {
    const resultsSection = document.getElementById('results-section');
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const explanationText = document.getElementById('explanation-text');
    const processingTime = document.getElementById('processing-time');

    // Show results section
    if (resultsSection) {
      resultsSection.classList.remove('hidden');
      resultsSection.classList.add('fade-in');
    }

    // Set verdict
    const verdict = data.verdict || 'INSUFFICIENT_INFO';
    if (verdictBadge) {
      verdictBadge.textContent = this.formatVerdict(verdict);
      verdictBadge.className = `verdict-badge ${this.getVerdictClass(verdict)}`;
    }

    // Set confidence
    const confidence = Math.round((data.confidence || 0) * 100);
    if (confidenceFill) {
      setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
      }, 300);
    }
    if (confidenceText) {
      confidenceText.textContent = `${confidence}%`;
    }

    // Set explanation
    if (explanationText) {
      explanationText.textContent = data.explanation || 'No explanation provided.';
    }

    // Set processing time
    if (processingTime && data.processing_time) {
      processingTime.textContent = `Processed in ${data.processing_time.toFixed(2)}s`;
    }

    // Update risk indicators
    this.updateRiskIndicators(data);

    // Handle Gemini comparison
    if (data.gemini_verdict) {
      const geminiComparison = document.getElementById('gemini-comparison');
      const geminiText = document.getElementById('gemini-text');
      
      if (geminiComparison && geminiText) {
        geminiText.textContent = data.gemini_verdict;
        geminiComparison.classList.remove('hidden');
      }
    }
  }

  /**
   * Display evidence
   */
  displayEvidence(data) {
    const evidenceSection = document.getElementById('evidence-section');
    const evidenceGrid = document.getElementById('evidence-grid');
    const evidenceCount = document.getElementById('evidence-count');

    if (!data.evidence || data.evidence.length === 0) {
      if (evidenceSection) {
        evidenceSection.classList.add('hidden');
      }
      return;
    }

    // Show evidence section
    if (evidenceSection) {
      evidenceSection.classList.remove('hidden');
      evidenceSection.classList.add('fade-in');
    }

    // Set evidence count
    if (evidenceCount) {
      evidenceCount.textContent = `${data.evidence.length} sources analyzed`;
    }

    // Clear existing evidence
    if (evidenceGrid) {
      evidenceGrid.innerHTML = '';

      // Create evidence cards
      data.evidence.forEach((evidence, index) => {
        const card = this.createEvidenceCard(evidence, index);
        evidenceGrid.appendChild(card);
      });
    }
  }

  /**
   * Create evidence card
   */
  createEvidenceCard(evidence, index) {
    const card = document.createElement('div');
    card.className = 'evidence-card slide-in';
    card.style.animationDelay = `${index * 0.1}s`;

    const sourceClass = this.getSourceClass(evidence.source || 'unknown');
    const credibilityScore = evidence.credibility_score ? 
      Math.round(evidence.credibility_score * 100) : 'N/A';
    const similarityScore = evidence.similarity_score ? 
      Math.round(evidence.similarity_score * 100) : 'N/A';

    card.innerHTML = `
      <div class="evidence-source ${sourceClass}">
        ${(evidence.source || 'unknown').toUpperCase()}
      </div>
      <h4 class="evidence-title">${evidence.title || 'No title available'}</h4>
      <p class="evidence-snippet">${evidence.snippet || 'No snippet available.'}</p>
      <div class="evidence-scores">
        <span class="score-badge">Credibility: ${credibilityScore}%</span>
        <span class="score-badge">Relevance: ${similarityScore}%</span>
      </div>
      ${evidence.url ? `
        <a href="${evidence.url}" target="_blank" class="evidence-url" rel="noopener noreferrer">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
            <polyline points="15,3 21,3 21,9"/>
            <line x1="10" y1="14" x2="21" y2="3"/>
          </svg>
          View Source
        </a>
      ` : ''}
    `;

    // Add click handler for expansion
    card.addEventListener('click', (e) => {
      if (e.target.tagName !== 'A') {
        card.classList.toggle('expanded');
      }
    });

    return card;
  }

  /**
   * Display intervention recommendations
   */
  displayInterventions(data) {
    const interventionSection = document.getElementById('intervention-section');
    const interventionList = document.getElementById('intervention-list');

    if (!data.interventions || data.interventions.length === 0) {
      if (interventionSection) {
        interventionSection.classList.add('hidden');
      }
      return;
    }

    if (interventionSection) {
      interventionSection.classList.remove('hidden');
    }

    if (interventionList) {
      interventionList.innerHTML = data.interventions.map(intervention => `
        <div class="intervention-item">
          <div class="intervention-type">${intervention.type}</div>
          <div class="intervention-description">${intervention.description}</div>
          <div class="intervention-urgency urgency-${intervention.urgency}">
            ${intervention.urgency.toUpperCase()}
          </div>
        </div>
      `).join('');
    }
  }

  /**
   * Update risk indicators
   */
  updateRiskIndicators(data) {
    const viralRisk = document.getElementById('viral-risk');
    const harmRisk = document.getElementById('harm-risk');

    if (viralRisk && data.viral_potential) {
      const viralPercentage = Math.round(data.viral_potential * 100);
      viralRisk.style.background = `conic-gradient(
        var(--color-danger) ${viralPercentage * 3.6}deg, 
        var(--color-border) ${viralPercentage * 3.6}deg
      )`;
    }

    if (harmRisk && data.harm_level) {
      const harmPercentage = Math.round(data.harm_level * 100);
      harmRisk.style.background = `conic-gradient(
        var(--color-warning) ${harmPercentage * 3.6}deg, 
        var(--color-border) ${harmPercentage * 3.6}deg
      )`;
    }
  }

  /**
   * Format verdict text
   */
  formatVerdict(verdict) {
    const verdictMap = {
      'TRUE': 'TRUE',
      'FALSE': 'FALSE',
      'CONFLICTING_EVIDENCE': 'CONFLICTING',
      'PARTIALLY_TRUE': 'PARTIAL',
      'INSUFFICIENT_INFO': 'INSUFFICIENT'
    };
    return verdictMap[verdict] || verdict.replace('_', ' ');
  }

  /**
   * Get verdict CSS class
   */
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

  /**
   * Get source CSS class
   */
  getSourceClass(source) {
    return `source-${source.toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
  }

  /**
   * Scroll to results
   */
  scrollToResults() {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
      setTimeout(() => {
        resultsSection.scrollIntoView({ 
          behavior: 'smooth',
          block: 'start'
        });
      }, 300);
    }
  }

  /**
   * Save draft
   */
  saveDraft() {
    const claimInput = document.getElementById('claim-input');
    if (!claimInput || !claimInput.value.trim()) {
      this.showErrorNotification('No content to save');
      return;
    }

    const draft = {
      id: Date.now(),
      content: claimInput.value.trim(),
      timestamp: new Date().toISOString(),
      settings: this.getFormData()
    };

    if (!this.drafts) {
      this.drafts = [];
    }

    this.drafts.unshift(draft);
    
    // Keep only last 10 drafts
    if (this.drafts.length > 10) {
      this.drafts = this.drafts.slice(0, 10);
    }

    localStorage.setItem('crisisguard-drafts', JSON.stringify(this.drafts));
    this.showSuccessNotification('Draft saved successfully');
    this.addActivityFeedItem('system', 'Draft saved');
  }

  /**
   * Share results
   */
  async shareResults() {
    try {
      const resultsData = this.getResultsData();
      
      if (navigator.share) {
        await navigator.share({
          title: 'Myth Munch AI - Fact Check Results',
          text: `Claim verification: ${resultsData.verdict}`,
          url: window.location.href
        });
      } else {
        // Fallback to clipboard
        const shareText = this.generateShareText(resultsData);
        await navigator.clipboard.writeText(shareText);
        this.showSuccessNotification('Results copied to clipboard');
      }

      this.addActivityFeedItem('system', 'Results shared');
    } catch (error) {
      console.error('Share failed:', error);
      this.showErrorNotification('Failed to share results');
    }
  }

  /**
   * Export results
   */
  exportResults() {
    try {
      const resultsData = this.getResultsData();
      const exportData = {
        timestamp: new Date().toISOString(),
        version: '2.1.1',
        ...resultsData
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `crisisguard-results-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      this.showSuccessNotification('Results exported successfully');
      this.addActivityFeedItem('system', 'Results exported');
    } catch (error) {
      console.error('Export failed:', error);
      this.showErrorNotification('Failed to export results');
    }
  }

  /**
   * Get current results data
   */
  getResultsData() {
    const verdictBadge = document.getElementById('verdict-badge');
    const confidenceText = document.getElementById('confidence-text');
    const explanationText = document.getElementById('explanation-text');
    const claimInput = document.getElementById('claim-input');

    return {
      claim: claimInput?.value || '',
      verdict: verdictBadge?.textContent || '',
      confidence: confidenceText?.textContent || '',
      explanation: explanationText?.textContent || '',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate share text
   */
  generateShareText(data) {
    return `
Myth Munch AI - Fact Check Results

Claim: ${data.claim}
Verdict: ${data.verdict}
Confidence: ${data.confidence}

${data.explanation}

Analyzed on ${new Date(data.timestamp).toLocaleString()}
    `.trim();
  }

  /**
   * Start monitoring
   */
  async startMonitoring() {
    try {
      const keywords = ['ukraine', 'covid', 'vaccine', 'climate change', 'election fraud'];
      
      await this.makeAPICall(this.apiConfig.endpoints.startMonitoring, {
        method: 'POST',
        body: JSON.stringify(keywords)
      });

      this.state.isMonitoring = true;
      this.updateMonitoringUI(true);
      this.showSuccessNotification('Monitoring started successfully');
      this.addActivityFeedItem('monitoring', 'Real-time monitoring started');
      
      // Start updates
      this.startMonitoringUpdates();
    } catch (error) {
      console.error('Failed to start monitoring:', error);
      this.showErrorNotification('Failed to start monitoring');
    }
  }

  /**
   * Stop monitoring
   */
  async stopMonitoring() {
    try {
      await this.makeAPICall(this.apiConfig.endpoints.stopMonitoring, {
        method: 'POST'
      });

      this.state.isMonitoring = false;
      this.updateMonitoringUI(false);
      this.showSuccessNotification('Monitoring stopped');
      this.addActivityFeedItem('monitoring', 'Real-time monitoring stopped');
      
      // Stop updates
      this.stopMonitoringUpdates();
    } catch (error) {
      console.error('Failed to stop monitoring:', error);
      this.showErrorNotification('Failed to stop monitoring');
    }
  }

  /**
   * Update monitoring UI
   */
  updateMonitoringUI(isMonitoring) {
    const startBtn = document.getElementById('start-monitoring');
    const stopBtn = document.getElementById('stop-monitoring');

    if (startBtn) {
      startBtn.disabled = isMonitoring;
    }
    if (stopBtn) {
      stopBtn.disabled = !isMonitoring;
    }
  }

  /**
   * Start monitoring updates
   */
  startMonitoringUpdates() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }

    this.monitoringInterval = setInterval(async () => {
      await this.updateAlerts();
      await this.updateTrends();
      this.updateRealtimeChart();
    }, 10000); // Every 10 seconds
  }

  /**
   * Stop monitoring updates
   */
  stopMonitoringUpdates() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
  }
  
  /**
   * Update alerts
   */
  async updateAlerts() {
    try {
      const data = await this.makeAPICall(this.apiConfig.endpoints.alerts);
      
      if (data && data.alerts) {
        this.state.alertsData = data.alerts;
        this.displayAlerts();
        this.updateAlertStats();
      }
    } catch (error) {
      console.error('Failed to update alerts:', error);
    }
  }

  /**
   * Update trends
   */
  async updateTrends() {
    try {
      const data = await this.makeAPICall(this.apiConfig.endpoints.trends);
      
      if (data && data.trending_keywords) {
        this.state.trendsData = data.trending_keywords;
        this.displayTrends();
        this.updateTrendStats();
      }
    } catch (error) {
      console.error('Failed to update trends:', error);
    }
  }

  /**
   * Display alerts
   */
  displayAlerts() {
    const container = document.getElementById('alerts-container');
    if (!container) return;

    if (!this.state.alertsData || this.state.alertsData.length === 0) {
      container.innerHTML = `
        <div class="no-data">
          <svg class="no-data-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.35-4.35"/>
          </svg>
          <p>No active alerts</p>
          <small>System monitoring for suspicious patterns</small>
        </div>
      `;
      return;
    }

    container.innerHTML = this.state.alertsData.map(alert => `
      <div class="alert-item">
        <div class="alert-header">
          <span class="alert-keyword">${alert.keyword || 'Unknown'}</span>
          <span class="alert-severity severity-${alert.severity || 'medium'}">
            ${(alert.severity || 'medium').toUpperCase()}
          </span>
        </div>
        <div class="alert-content">
          ${alert.description || 'Suspicious pattern detected'}
        </div>
        <div class="alert-footer">
          <span class="alert-time">${this.formatTime(alert.timestamp)}</span>
          <span class="alert-confidence">Confidence: ${Math.round((alert.confidence || 0.8) * 100)}%</span>
        </div>
      </div>
    `).join('');
  }

  /**
   * Display trends
   */
  displayTrends() {
    const container = document.getElementById('trends-container');
    if (!container) return;

    if (!this.state.trendsData || this.state.trendsData.length === 0) {
      container.innerHTML = `
        <div class="no-data">
          <svg class="no-data-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>
          </svg>
          <p>No trending topics</p>
          <small>Start monitoring to see trends</small>
        </div>
      `;
      return;
    }

    const maxMentions = Math.max(...this.state.trendsData.map(([,count]) => count));

    container.innerHTML = this.state.trendsData.map(([keyword, mentions]) => {
      const percentage = maxMentions > 0 ? (mentions / maxMentions * 100) : 0;
      const change = (Math.random() - 0.5) * 20; // Simulated change

      return `
        <div class="trend-item">
          <div class="trend-info">
            <div class="trend-keyword">${keyword}</div>
            <div class="trend-mentions">${mentions} mentions</div>
          </div>
          <div class="trend-visual">
            <div class="trend-bar">
              <div class="trend-fill" style="width: ${percentage}%"></div>
            </div>
            <span class="trend-change ${change >= 0 ? 'positive' : 'negative'}">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                ${change >= 0 ? 
                  '<polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/>' : 
                  '<polyline points="1,6 8.5,13.5 13.5,8.5 23,18"/>'
                }
              </svg>
              ${Math.abs(change).toFixed(1)}%
            </span>
          </div>
        </div>
      `;
    }).join('');
  }

  /**
   * Update dashboard stats
   */
  updateDashboardStats() {
    // Update stats with current data
    const activeAlerts = document.getElementById('active-alerts');
    const trendMentions = document.getElementById('trend-mentions');
    const factsChecked = document.getElementById('facts-checked');

    if (activeAlerts) {
      activeAlerts.textContent = this.state.alertsData?.length || 0;
    }

    if (trendMentions) {
      const totalMentions = this.state.trendsData?.reduce((sum, [,count]) => sum + count, 0) || 0;
      trendMentions.textContent = Math.round(totalMentions / 24); // Per hour estimate
    }

    if (factsChecked) {
      const current = parseInt(factsChecked.textContent) || 45782;
      factsChecked.textContent = current + Math.floor(Math.random() * 5);
    }
  }

  /**
   * Update real-time chart
   */
  updateRealtimeChart() {
    const chart = this.state.chartInstances.realtime;
    if (!chart) return;

    const now = new Date();
    const timeLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const value = Math.floor(Math.random() * 100) + 50;

    chart.data.labels.push(timeLabel);
    chart.data.datasets[0].data.push(value);

    // Keep only last 20 data points
    if (chart.data.labels.length > 20) {
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }

    chart.update('none');
  }

  /**
   * Update connection status
   */
  updateConnectionStatus(status) {
    this.state.connectionStatus = status;
    
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');

    if (statusDot) {
      statusDot.className = `status-dot ${status === 'connected' ? 'connected' : 'disconnected'}`;
    }

    if (statusText) {
      statusText.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
    }
  }

  /**
   * Check connection status
   */
  async checkConnectionStatus() {
    try {
      await this.makeAPICall(this.apiConfig.endpoints.health);
      this.updateConnectionStatus('connected');
    } catch (error) {
      this.updateConnectionStatus('disconnected');
    }
  }

  /**
   * Add activity feed item
   */
  addActivityFeedItem(type, message) {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;

    const item = {
      type,
      message,
      timestamp: new Date().toISOString(),
      id: Date.now()
    };

    this.state.activityFeed.unshift(item);

    // Keep only last 50 items
    if (this.state.activityFeed.length > 50) {
      this.state.activityFeed = this.state.activityFeed.slice(0, 50);
    }

    this.renderActivityFeed();
  }

  /**
   * Render activity feed
   */
  renderActivityFeed() {
    const feed = document.getElementById('activity-feed');
    if (!feed) return;

    feed.innerHTML = this.state.activityFeed.slice(0, 20).map(item => `
      <div class="activity-item">
        <div class="activity-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${this.getActivityIconSVG(item.type)}
          </svg>
        </div>
        <div class="activity-content">
          <div class="activity-message">${item.message}</div>
          <div class="activity-time">${this.formatTime(item.timestamp)}</div>
        </div>
      </div>
    `).join('');
  }

  /**
   * Get activity icon SVG
   */
  getActivityIconSVG(type) {
    const icons = {
      'system': '<circle cx="12" cy="12" r="3"/><path d="m12 1 2.09 3.26L18 6l-3.69 2.39L15 13l-3-2-3 2 .69-4.61L6 6l3.91-1.74L12 1z"/>',
      'analysis': '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',
      'monitoring': '<polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>',
      'input': '<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>',
      'navigation': '<circle cx="12" cy="12" r="10"/><polygon points="16.24,7.76 14.12,14.12 7.76,16.24 9.88,9.88"/>',
      'error': '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
      'emergency': '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
    };
    return icons[type] || icons['system'];
  }

  /**
   * Format timestamp
   */
  formatTime(timestamp) {
    if (!timestamp) return 'Unknown time';
    
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diff = now - date;

      if (diff < 60000) { // Less than 1 minute
        return 'Just now';
      } else if (diff < 3600000) { // Less than 1 hour
        return `${Math.floor(diff / 60000)}m ago`;
      } else if (diff < 86400000) { // Less than 1 day
        return `${Math.floor(diff / 3600000)}h ago`;
      } else {
        return date.toLocaleDateString();
      }
    } catch (error) {
      return 'Unknown time';
    }
  }

  /**
   * Refresh dashboard
   */
  async refreshDashboard() {
    try {
      this.showSuccessNotification('Refreshing dashboard...');
      
      await Promise.all([
        this.updateAlerts(),
        this.updateTrends(),
        this.checkSystemHealth()
      ]);

      this.updateDashboardStats();
      this.addActivityFeedItem('system', 'Dashboard refreshed');
    } catch (error) {
      console.error('Dashboard refresh failed:', error);
      this.showErrorNotification('Failed to refresh dashboard');
    }
  }

  /**
   * Show error notification
   */
  showErrorNotification(message) {
    this.showNotification(message, 'error');
  }

  /**
   * Show success notification
   */
  showSuccessNotification(message) {
    this.showNotification(message, 'success');
  }

  /**
   * Show notification
   */
  showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    const iconSVG = {
      'error': '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
      'success': '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22,4 12,14.01 9,11.01"/>',
      'info': '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
    };

    notification.innerHTML = `
      <div class="notification-content">
        <svg class="notification-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          ${iconSVG[type] || iconSVG['info']}
        </svg>
        <span class="notification-message">${message}</span>
      </div>
      <button class="notification-close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    `;

    // Add to page
    document.body.appendChild(notification);

    // Position notification
    const notifications = document.querySelectorAll('.notification');
    notification.style.top = `${20 + (notifications.length - 1) * 80}px`;
    notification.style.right = '20px';

    // Auto remove
    setTimeout(() => {
      if (notification.parentNode) {
        notification.remove();
      }
    }, 5000);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
      notification.remove();
    });
  }

  /**
   * Show error message
   */
  showError(message) {
    console.error('Application error:', message);
    this.showErrorNotification(`Error: ${message}`);
    this.addActivityFeedItem('error', message);
  }

  /**
   * Update stats counters
   */
  updateStats() {
    const totalChecks = document.getElementById('total-checks');
    if (totalChecks) {
      const current = parseInt(totalChecks.textContent.replace(/,/g, '')) || 45782;
      totalChecks.textContent = (current + 1).toLocaleString();
    }
  }
}

// Additional CSS for notifications (inject dynamically)
const notificationStyles = `
  .notification {
    position: fixed;
    z-index: 10000;
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: var(--space-md);
    min-width: 300px;
    max-width: 400px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-sm);
    animation: slideInRight 0.3s ease-out;
    box-shadow: var(--shadow-lg);
  }

  .notification-error {
    border-left: 4px solid var(--color-danger);
  }

  .notification-success {
    border-left: 4px solid var(--color-success);
  }

  .notification-info {
    border-left: 4px solid var(--color-info);
  }

  .notification-content {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    flex: 1;
  }

  .notification-icon {
    flex-shrink: 0;
  }

  .notification-message {
    font-size: 0.875rem;
    color: var(--color-text-primary);
  }

  .notification-close {
    background: none;
    border: none;
    color: var(--color-text-secondary);
    cursor: pointer;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .notification-close:hover {
    color: var(--color-text-primary);
  }

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

  .animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: .5;
    }
  }
`;

// Inject notification styles
if (!document.getElementById('notification-styles')) {
  const style = document.createElement('style');
  style.id = 'notification-styles';
  style.textContent = notificationStyles;
  document.head.appendChild(style);
}
const infoContent = {
  privacy: {
    title: "Privacy Policy",
    body: `<p>Myth Munch AI values your privacy. We collect, use, and protect information in accordance with applicable data protection laws.</p>
           <ul>
             <li><strong>Information Collected :</strong> We may collect data you input to verify claims, logs related to system usage, and technical details about your device and browser. Where possible, inputs are anonymized and stored securely.</li>
             <li><strong>Use of Information :</strong> Data may be used to improve service accuracy, monitor system health, and generate aggregate statistics. Personal information is never sold or used for unrelated marketing.</li>
             <li><strong>AI Data Handling :</strong> Inputs and outputs processed by our AI models may be stored temporarily for system improvement. No personally identifiable information is shared with third parties except as required by law.  </li>
             <li><strong>User Rights :</strong> You may access, correct, or delete personal information stored about you by contacting support.  </li>
             <li><strong>Security Measures :</strong> We use encryption, access controls, and regular audits to protect your data.  </li>
             <li><strong>AI-Specific Disclaimers :</strong> AI systems may produce inaccurate or incomplete results. All AI outputs should be used as guidance, not as definitive fact.  </li>
             <li><strong>Contact :</strong> For privacy questions or to exercise your rights, email adithkp03@gmail.com.  </li>
           </ul>`
  },
  terms: {
    title: "Terms of Service",
    body: `<p>These Terms of Service govern your use of Myth Munch AI.</p>
           <ul>
             <li><strong>Service Description :</strong>  Myth Munch AI provides automated verification and evidence aggregation for public claims using AI technologies.</li>
             <li><strong>Acceptable Use :</strong> Do not use the service to submit illegal, harmful, or copyrighted material, nor to automate credit, insurance, employment, or similarly critical decisions.</li>
             <li><strong>AI Limitations :</strong> AI-generated outputs should be independently verified before acting on them. CrisisGuard AI is not liable for outcomes based on system analysis.</li>
             <li><strong>Intellectual Property :</strong> Ownership of user-submitted content remains with you. Generated outputs may be used for improvement, analytics, and display within the system.</li>
             <li><strong>Service availability :</strong> The service may be interrupted for maintenance or unforeseen reasons. Service is provided as-is with no uptime guarantees.</li>
             <li><strong>Liability :</strong> CrisisGuard AI is not responsible for losses arising from use of the system, AI model errors, or third-party data sources.</li>
             <li><strong>Changes :</strong> Terms may be updated periodically. Continued use indicates agreement with updated terms.</li>
             <li><strong>Governing Law :</strong> These terms are subject to the laws of your local jurisdiction.</li>
             <li><strong>Contact :</strong>For queries about these terms, contact adithkp03@gmail.com.</li>
           </ul>`
  },
  api: {
    title: "API Documentation",
    body: `<p>Welcome to the Myth Munch AI API Docs.</p>
           <ul>
             <li><strong>/fact-check</strong> (POST): Submit claim for verification.</li>
             <li><strong>/trends</strong> (GET): Get trending topics.</li>
             <li><strong>API Key Required</strong> via Authorization header.</li>
             <li>Sample Request/Response available via adithkp03@gmail.com.</li>
           </ul>`
  },
  support: {
    title: "Support",
    body: `<p>Need help? Our support team is ready to assist you.</p>
           <ul>
             <li><strong>Help Center:</strong> Guides and troubleshooting tips.</li>
             <li><strong>Email:</strong> support@crisisguard.ai</li>
             <li><strong>Community:</strong> Forums and feedback portal.</li>
           </ul>`
  }
};

document.querySelectorAll('.footer-link').forEach(link => {
  link.addEventListener('click', function(e) {
    e.preventDefault();
    const key = this.getAttribute('data-info');
    document.getElementById('modal-title').textContent = infoContent[key].title;
    document.getElementById('modal-body').innerHTML = infoContent[key].body;
    document.getElementById('info-modal').classList.remove('hidden');
  });
});

window.addEventListener('DOMContentLoaded', function() {
  // Attach footer links
  document.querySelectorAll('.footer-link').forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      const key = this.getAttribute('data-info');
      document.getElementById('modal-title').textContent = infoContent[key].title;
      document.getElementById('modal-body').innerHTML = infoContent[key].body;
      document.getElementById('info-modal').classList.remove('hidden');
    });
  });

  // Attach modal close button event using the new ID
  const closeBtn = document.getElementById('modal-close-btn');
  if (closeBtn) {
      closeBtn.addEventListener('click', function() {
      document.getElementById('info-modal').classList.add('hidden');
    });
  }

  // ESC key closes modal
  window.addEventListener('keydown', function(e) {
    if (e.key === "Escape") {
      document.getElementById('info-modal').classList.add('hidden');
    }
  });
});


// Export for use in other scripts if needed
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CrisisGuardApp;
}