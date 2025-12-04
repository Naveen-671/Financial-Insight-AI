import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Shield, FileText, CheckCircle2, AlertCircle, Download, Tag } from 'lucide-react';
import { cn } from '../../lib/utils';

export default function ClauseExtraction({ data }) {
  // Process API data
  const processedData = useMemo(() => {
    console.log('ClauseExtraction received data:', data);

    if (!data || (!data.clauses && !data.results && !data.langextract)) {
      return {
        clauses: [],
        totalClauses: 0
      };
    }

    let clauses = [];

    // Handle backend format: results -> model_name -> extractions
    if (data.results) {
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
      totalClauses: clauses.length
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

  // Determine the correct visualization URL
  // Check root, then langextract, then results
  const visualizationUrl = data.highlighted_html_url ||
    (data.langextract && data.langextract.highlighted_html_url) ||
    (data.results && data.results['gemini-1.5-flash'] && data.results['gemini-1.5-flash'].json_output && data.results['gemini-1.5-flash'].json_output.highlighted_html_url);

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

      {/* Document Analysis Preview */}
      {visualizationUrl && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-8"
        >
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
            <FileText className="w-5 h-5 text-horizon-primary" />
            Document Analysis Preview
          </h3>
          <div className="w-full h-[500px] rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 bg-white">
            <iframe
              src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'}${visualizationUrl}`}
              title="Clause Analysis Visualization"
              className="w-full h-full"
              style={{ border: 'none' }}
            />
          </div>
        </motion.div>
      )}

      {/* Clauses List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
          <FileText className="w-5 h-5 text-horizon-primary" />
          Extracted Clauses
        </h3>
        <div className="space-y-4">
          {processedData.clauses.map((clause, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.1 }}
              className={cn("p-5 rounded-xl border transition-all hover:shadow-md", getImportanceColor(clause.importance))}
            >
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-3">
                <div className="flex items-center gap-2">
                  <Tag className="w-4 h-4 opacity-70" />
                  <h4 className="font-bold text-lg">{clause.type}</h4>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-white/50 dark:bg-black/20 backdrop-blur-sm self-start">
                  {clause.importance} Priority
                </span>
              </div>

              <div className="bg-white/60 dark:bg-black/20 p-4 rounded-lg mb-3">
                <p className="text-sm leading-relaxed font-mono text-gray-800 dark:text-gray-200">
                  "{clause.text}"
                </p>
              </div>

              {/* Attributes Display */}
              {clause.attributes && Object.keys(clause.attributes).length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-3 text-sm">
                  {Object.entries(clause.attributes).map(([key, value]) => (
                    key !== 'confidence' && (
                      <div key={key} className="flex items-center gap-2 text-gray-600 dark:text-gray-300">
                        <span className="font-semibold capitalize opacity-75">{key.replace(/_/g, ' ')}:</span>
                        <span>{String(value)}</span>
                      </div>
                    )
                  ))}
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </motion.div>

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
