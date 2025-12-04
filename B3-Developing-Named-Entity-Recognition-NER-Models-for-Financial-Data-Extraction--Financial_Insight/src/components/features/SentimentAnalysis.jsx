import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Smile, Frown, Meh, TrendingUp, BarChart3, FileText } from 'lucide-react';
import { cn } from '../../lib/utils';

export default function SentimentAnalysis({ data }) {
  // Process API data
  const processedData = useMemo(() => {
    // Strict check for real data
    if (!data || (!data.statistics && !data.sentence_results)) {
      return {
        overallSentiment: 'neutral',
        score: 0,
        confidence: 0,
        sentenceAnalysis: []
      };
    }

    // Handle Backend Response Format
    const sentences = data.sentence_results || [];
    const mappedSentences = sentences.map(s => ({
      text: s.text,
      sentiment: (s.label || 'neutral').toLowerCase(),
      score: s.score || 0
    }));

    // Calculate overall confidence/score from statistics if available
    const overallScore = data.statistics?.average_scores?.[data.statistics?.overall_sentiment] || 0;

    return {
      overallSentiment: (data.statistics?.overall_sentiment || 'neutral').toLowerCase(),
      score: overallScore,
      confidence: overallScore,
      sentenceAnalysis: mappedSentences
    };
  }, [data]);

  const getSentimentIcon = (sentiment) => {
    if (sentiment === 'positive') return <Smile className="w-6 h-6 text-green-500" />;
    if (sentiment === 'negative') return <Frown className="w-6 h-6 text-red-500" />;
    return <Meh className="w-6 h-6 text-gray-500" />;
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'positive') return 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400';
    if (sentiment === 'negative') return 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400';
    return 'bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400';
  };

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center text-white shadow-lg">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Sentiment Analysis</h2>
            <p className="text-gray-600 dark:text-gray-400">FinBERT-powered sentiment detection</p>
          </div>
        </div>
      </motion.div>

      {/* Overall Sentiment Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6">Overall Sentiment</h3>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {getSentimentIcon(processedData.overallSentiment)}
            <div>
              <div className={cn("text-2xl font-bold capitalize", getSentimentColor(processedData.overallSentiment))}>
                {processedData.overallSentiment}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Score: {(processedData.score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-gray-900 dark:text-white">
              {(processedData.confidence * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Confidence</div>
          </div>
        </div>
      </motion.div>

      {/* Document View */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
          <FileText className="w-5 h-5 text-horizon-primary" />
          Document Analysis Preview
        </h3>

        <div className="bg-gray-50/50 dark:bg-horizon-secondary/30 rounded-xl p-6 text-sm text-gray-700 dark:text-gray-300 leading-relaxed space-y-4 max-h-[600px] overflow-y-auto">
          {data && data.highlighted_html_url ? (
            <div className="w-full h-[500px] bg-white rounded-lg overflow-hidden border border-gray-200 dark:border-white/10">
              <iframe
                src={`http://localhost:8001${data.highlighted_html_url}`}
                className="w-full h-full"
                title="Sentiment Visualization"
              />
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              <p>Document preview not available.</p>
              <p className="text-xs mt-2">Try processing a new document to generate the preview.</p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Sentence-level Analysis */}
      {processedData.sentenceAnalysis.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-8"
        >
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-horizon-primary" />
            Sentence-Level Breakdown
          </h3>
          <div className="space-y-3">
            {processedData.sentenceAnalysis.map((item, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={cn("p-4 rounded-lg", getSentimentColor(item.sentiment))}
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="flex-1">{item.text}</p>
                  <div className="flex items-center gap-2">
                    {getSentimentIcon(item.sentiment)}
                    <span className="font-semibold">{(item.score * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
