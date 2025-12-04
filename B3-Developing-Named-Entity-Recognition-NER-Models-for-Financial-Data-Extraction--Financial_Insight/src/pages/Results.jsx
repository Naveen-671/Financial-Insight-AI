import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  TrendingUp,
  PieChart,
  Download,
  Building,
  DollarSign,
  Calendar,
  BarChart3,
  Target,
  CheckCircle2,
  Sparkles,
  Brain,
  Smile,
  FileText,
  Zap,
  FileDown,
  Loader2
} from 'lucide-react'
import { cn } from '../lib/utils'
import NERAnalysis from '../components/features/NERAnalysis'
import SentimentAnalysis from '../components/features/SentimentAnalysis'
import ClauseExtraction from '../components/features/ClauseExtraction'
import AnalysisCharts from '../components/features/AnalysisCharts'
import { exportResults, generateReport, downloadBlob } from '../services/api'

export default function Results({ analysisResults: globalResults }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('ner');
  const [selectedAnalyses, setSelectedAnalyses] = useState({
    ner: true,
    sentiment: true,
    clauses: false
  });
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  // Load selected analyses from navigation state or sessionStorage
  useEffect(() => {
    const data = location.state?.analysisResults || globalResults;
    if (data) {
      setAnalysisResults(data);
      setSelectedAnalyses(location.state?.selectedFeatures || selectedAnalyses);
      setUploadedFiles(location.state?.fileName ? [location.state.fileName] : []);

      // Set initial tab based on selected analyses
      const features = location.state.selectedFeatures;
      if (features?.ner) setActiveTab('ner');
      else if (features?.sentiment) setActiveTab('sentiment');
      else if (features?.clauses) setActiveTab('clause');
    } else {
      // Fallback to sessionStorage for backwards compatibility
      const stored = sessionStorage.getItem('selectedAnalyses');
      const files = sessionStorage.getItem('uploadedFiles');
      if (stored) {
        setSelectedAnalyses(JSON.parse(stored));
        const analyses = JSON.parse(stored);
        if (analyses.ner) setActiveTab('ner');
        else if (analyses.sentiment) setActiveTab('sentiment');
        else if (analyses.clauses) setActiveTab('clause');
      }
      if (files) {
        setUploadedFiles(JSON.parse(files));
      }
    }
  }, [location]);

  const allTabs = [
    {
      id: 'ner',
      label: 'Named Entity Recognition',
      icon: Brain,
      description: 'Extract financial entities'
    },
    {
      id: 'sentiment',
      label: 'Sentiment Analysis',
      icon: Smile,
      description: 'Document sentiment classification'
    },
    {
      id: 'clause',
      label: 'Clause Extraction',
      icon: FileText,
      description: 'Contract clause identification'
    }
  ];

  // Filter tabs based on selected analyses
  const tabs = allTabs.filter(tab => {
    if (tab.id === 'ner') return selectedAnalyses.ner;
    if (tab.id === 'sentiment') return selectedAnalyses.sentiment;
    if (tab.id === 'clause') return selectedAnalyses.clauses;
    return false;
  });

  // Defensive check: If no analysis results, show loading or empty state
  if (!analysisResults) {
    return (
      <div className="min-h-screen pt-20 pb-12 px-4 sm:px-6 lg:px-8 bg-gray-50 dark:bg-horizon-primary/5 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-horizon-primary animate-spin mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Waiting for analysis results...</h2>
          <p className="text-gray-500 mt-2">If this takes too long, please try uploading again.</p>
          <button
            onClick={() => navigate('/upload')}
            className="mt-6 px-4 py-2 bg-horizon-primary text-white rounded-lg hover:bg-horizon-primary/90 transition-colors"
          >
            Return to Upload
          </button>
        </div>
      </div>
    );
  }

  const stats = [
    { label: 'Analysis Duration', value: '2.4s', icon: Zap },
    { label: 'Entities Extracted', value: '53', icon: Target },
    { label: 'Documents Processed', value: '1', icon: BarChart3 },
    { label: 'Overall Confidence', value: '94%', icon: TrendingUp },
  ];

  // Download handler
  const handleDownload = async (format = 'json') => {
    if (!analysisResults) return;

    setIsDownloading(true);
    try {
      const blob = await exportResults(analysisResults, format);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const filename = `financial_analysis_${timestamp}.${format}`;
      downloadBlob(blob, filename);
    } catch (error) {
      console.error('Download error:', error);
      alert(`Failed to download results: ${error.message}`);
    } finally {
      setIsDownloading(false);
    }
  };

  // Report generation handler
  const handleGenerateReport = async () => {
    if (!analysisResults) return;

    setIsGeneratingReport(true);
    try {
      const blob = await generateReport(analysisResults);
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const filename = `financial_report_${timestamp}.html`;
      downloadBlob(blob, filename);
    } catch (error) {
      console.error('Report generation error:', error);
      alert(`Failed to generate report: ${error.message}`);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  // Navigate to upload page
  const handleAnalyzeMore = () => {
    navigate('/upload');
  };

  return (
    <div className="min-h-screen bg-horizon-light dark:bg-horizon-dark transition-colors duration-300 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header Section */}
        <motion.div
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="mb-12"
        >
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 glass px-6 py-2 rounded-full text-sm mb-6 animate-pulse">
              <Sparkles className="w-4 h-4 text-horizon-accent.blue" />
              <span className="text-gray-700 dark:text-gray-200">AI Analysis Complete</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-4">
              <span className="gradient-text">Comprehensive Analysis Results</span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Advanced AI insights from your financial documents
            </p>
          </div>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12"
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={index}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3 + index * 0.1, type: 'spring' }}
                className="glass-card p-4 text-center group hover:scale-105 transition-transform"
              >
                <div className="w-10 h-10 bg-gradient-horizon rounded-xl flex items-center justify-center text-white mx-auto mb-2 group-hover:shadow-horizon transition-shadow">
                  <Icon className="w-5 h-5" />
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-white">{stat.value}</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">{stat.label}</div>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Visualizations */}
        <AnalysisCharts data={analysisResults} />

        {/* Tab Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="glass-card p-1 mb-8 inline-flex w-full gap-1 flex-wrap md:w-auto md:flex-nowrap"
        >
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <motion.button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  "flex-1 md:flex-none flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all duration-300 text-sm md:text-base",
                  activeTab === tab.id
                    ? "bg-gradient-horizon text-white shadow-horizon"
                    : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-horizon-secondary/50"
                )}
              >
                <Icon className="w-5 h-5" />
                <span>{tab.label}</span>
              </motion.button>
            );
          })}
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            className="mb-12"
          >
            {activeTab === 'ner' && <NERAnalysis data={analysisResults?.ner} />}
            {activeTab === 'sentiment' && <SentimentAnalysis data={analysisResults?.finbert} />}
            {activeTab === 'clause' && <ClauseExtraction data={analysisResults?.langextract} />}
          </motion.div>
        </AnimatePresence>

        {/* Footer Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="glass-card p-8 text-center"
        >
          <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Ready for More?
          </h3>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center flex-wrap">
            <button
              onClick={() => handleDownload('json')}
              disabled={isDownloading}
              className="flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-horizon text-white font-semibold shadow-horizon hover:shadow-horizon-lg transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isDownloading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Download className="w-5 h-5" />
              )}
              Download All Results
            </button>
            <button
              onClick={handleGenerateReport}
              disabled={isGeneratingReport}
              className="flex items-center gap-3 px-8 py-4 rounded-xl glass-card border border-gray-200/50 dark:border-white/10 text-gray-700 dark:text-gray-200 font-semibold hover:bg-gray-50 dark:hover:bg-horizon-secondary/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isGeneratingReport ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <BarChart3 className="w-5 h-5" />
              )}
              Generate Report
            </button>
            <button
              onClick={handleAnalyzeMore}
              className="flex items-center gap-3 px-8 py-4 rounded-xl glass-card border border-gray-200/50 dark:border-white/10 text-gray-700 dark:text-gray-200 font-semibold hover:bg-gray-50 dark:hover:bg-horizon-secondary/50 transition-all duration-300"
            >
              <Zap className="w-5 h-5" />
              Analyze More Documents
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
