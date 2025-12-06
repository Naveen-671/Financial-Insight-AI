import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Shield, FileText, CheckCircle2, AlertCircle, Download, Tag } from 'lucide-react';
import { cn } from '../../lib/utils';
import InteractiveClauseViewer from './InteractiveClauseViewer';

export default function ClauseExtraction({ data }) {
  // Process API data
  const processedData = useMemo(() => {
    console.log('ClauseExtraction received data:', data);

    if (!data || (!data.clauses && !data.results && !data.langextract)) {
      return {
        clauses: [],
        totalClauses: 0,
        text: ""
      };
    }

    let clauses = [];
    let fullText = data.text || ""; // Try to get text from root

    // Handle backend format: results -> model_name -> extractions
    if (data.results) {
      // Try getting text from nested structure if not at root
      const firstResult = Object.values(data.results)[0];
      if (!fullText && firstResult?.json_output?.text) {
        fullText = firstResult.json_output.text;
      }

      Object.values(data.results).forEach(modelResult => {
        if (modelResult.extractions && Array.isArray(modelResult.extractions)) {
          clauses = [...clauses, ...modelResult.extractions];
        }
      });
    } else if (data.clauses && Array.isArray(data.clauses)) {
      clauses = data.clauses;
    } else if (data.langextract && data.langextract.clauses) {
      clauses = data.langextract.clauses;
    }

    // Fallback: if we still don't have text but have clauses, we might be in a legacy state. 
    // Ideally backend should provide text. If not, the viewer will show "No text content".

    // Map to standardized format with type-based importance
    clauses = clauses.map(ext => {
      const type = (ext.extraction_class || ext.type || ext.class || 'General Clause').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      const typeLower = type.toLowerCase();

      // Determine importance based on clause type
      let importance = 'medium';
      if (typeLower.includes('liability') || typeLower.includes('termination') || typeLower.includes('indemnity') || typeLower.includes('payment') || typeLower.includes('dispute')) {
        importance = 'high';
      } else if (typeLower.includes('confidentiality') || typeLower.includes('warranty') || typeLower.includes('governing') || typeLower.includes('interest')) {
        importance = 'medium';
      } else {
        importance = 'low';
      }

      return {
        type: type,
        text: ext.extraction_text || ext.text || 'No text available',
        importance: importance,
        confidence: ext.attributes?.confidence || 'medium',
        attributes: ext.attributes || {}
      };
    });

    // Filter out invalid/null clauses
    clauses = clauses.filter(c =>
      c.text &&
      c.text.toLowerCase() !== 'null' &&
      c.text !== 'No text available' &&
      c.text.length > 5
    );

    console.log('Processed clauses:', clauses);

    return {
      clauses: clauses,
      totalClauses: clauses.length,
      text: fullText
    };
  }, [data]);

  const getImportanceColor = (importance) => {
    if (importance === 'high') return 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800';
    if (importance === 'medium') return 'bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-400 border-yellow-200 dark:border-yellow-800';
    return 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800';
  };

  const handleDownload = () => {
    if (processedData.clauses.length === 0) return;

    const jsonString = JSON.stringify(processedData.clauses, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'extracted_clauses.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center text-white shadow-lg">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Clause Extraction</h2>
            <p className="text-gray-600 dark:text-gray-400">AI-powered legal clause identification</p>
          </div>
        </div>

        {processedData.totalClauses > 0 && (
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-sm font-medium text-gray-700 dark:text-gray-300 shadow-sm"
          >
            <Download className="w-4 h-4" />
            Download Report
          </button>
        )}
      </motion.div>

      {/* Summary Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-gray-50 dark:bg-gray-800/50 rounded-xl">
            <div className="text-3xl font-bold text-gray-900 dark:text-white">{processedData.totalClauses}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Clauses</div>
          </div>
          <div className="text-center p-4 bg-red-50 dark:bg-red-900/10 rounded-xl">
            <div className="text-3xl font-bold text-red-600 dark:text-red-400">
              {processedData.clauses.filter(c => c.importance === 'high').length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">High Priority</div>
          </div>
          <div className="text-center p-4 bg-yellow-50 dark:bg-yellow-900/10 rounded-xl">
            <div className="text-3xl font-bold text-yellow-600 dark:text-yellow-400">
              {processedData.clauses.filter(c => c.importance === 'medium').length}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Medium Priority</div>
          </div>
        </div>
      </motion.div>

      {/* Interactive Visualization */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className=""
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
          <FileText className="w-5 h-5 text-horizon-primary" />
          Analysis Visualization
        </h3>

        <InteractiveClauseViewer text={processedData.text} clauses={processedData.clauses} />

      </motion.div>

      {/* Detailed List (Optional fallback or extra view) */}
      <div className="mt-8">
        <h4 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-4 opacity-80">All Extractions Table</h4>
        <div className="space-y-2 opacity-70 hover:opacity-100 transition-opacity">
          {processedData.clauses.map((clause, idx) => (
            <div key={idx} className="flex gap-4 text-sm border-b border-gray-100 dark:border-gray-800 py-2">
              <span className="font-semibold w-1/3 truncate text-horizon-primary">{clause.type}</span>
              <span className="w-2/3 truncate text-gray-600 dark:text-gray-400">{clause.text}</span>
            </div>
          ))}
        </div>
      </div>

      {processedData.totalClauses === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-card p-12 text-center"
        >
          <div className="w-16 h-16 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">No Clauses Found</h3>
          <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
            We couldn't identify any specific legal or financial clauses in this document. Try uploading a contract or agreement file.
          </p>
        </motion.div>
      )}
    </div>
  );
}
