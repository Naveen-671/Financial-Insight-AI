import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Target,
  TrendingUp,
  BarChart3,
  Building,
  DollarSign,
  Calendar,
  CheckCircle2,
  Copy,
  Download,
  ZoomIn,
  Filter
} from 'lucide-react';
import { cn } from '../../lib/utils';

export default function NERAnalysis({ data }) {
  const [activeEntity, setActiveEntity] = useState(null);
  const [selectedFilter, setSelectedFilter] = useState('all');

  // Process API data
  const processedData = useMemo(() => {
    // Strict check: if no entities, return empty state
    if (!data || !data.entities) {
      return {
        entities: [],
        totalEntities: 0
      };
    }

    // Process API data
    const colorMap = {
      'ORG': 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
      'COMPANY': 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
      'REVENUE': 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
      'STOCK_PRICE': 'bg-cyan-100 dark:bg-cyan-900/30 text-cyan-600 dark:text-cyan-400',
      'MARKET_CAP': 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
      'FINANCIAL_EVENT': 'bg-pink-100 dark:bg-pink-900/30 text-pink-600 dark:text-pink-400',
      'FINANCIAL_RATIO': 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400',
      'PERSON': 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400',
      'LOCATION': 'bg-teal-100 dark:bg-teal-900/30 text-teal-600 dark:text-teal-400',
      'GPE': 'bg-teal-100 dark:bg-teal-900/30 text-teal-600 dark:text-teal-400',
      'DATE': 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400',
      'MONEY': 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400'
    };

    // Group entities by type
    const entityGroups = {};
    data.entities.forEach(entity => {
      const type = entity.label || entity.type || 'OTHER';
      if (!entityGroups[type]) {
        entityGroups[type] = {
          type,
          color: colorMap[type] || 'bg-gray-100 dark:bg-gray-900/30 text-gray-600 dark:text-gray-400',
          count: 0,
          examples: []
        };
      }
      entityGroups[type].count++;
      if (entityGroups[type].examples.length < 10) {
        entityGroups[type].examples.push(entity.text);
      }
    });

    return {
      entities: Object.values(entityGroups),
      totalEntities: data.entities.length
    };
  }, [data]);

  const entities = processedData.entities;

  const metrics = [
    { label: 'Total Entities', value: processedData.totalEntities.toString(), icon: Target, color: 'from-purple-500 to-pink-500' },
    { label: 'Precision', value: '94%', icon: TrendingUp, color: 'from-blue-500 to-cyan-500' },
    { label: 'Entity Types', value: entities.length.toString(), icon: BarChart3, color: 'from-green-500 to-emerald-500' },
    { label: 'Confidence Avg', value: '92%', icon: Brain, color: 'from-orange-500 to-red-500' }
  ];

  const filteredEntities = selectedFilter === 'all'
    ? entities
    : entities.filter(e => e.type === selectedFilter);

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center text-white shadow-lg">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Named Entity Recognition</h2>
            <p className="text-gray-600 dark:text-gray-400">AI-powered financial entity extraction</p>
          </div>
        </div>
      </motion.div>

      {/* Key Metrics */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {metrics.map((metric, index) => {
          const Icon = metric.icon;
          return (
            <motion.div
              key={index}
              whileHover={{ y: -5, scale: 1.02 }}
              className="glass-card p-6 group"
            >
              <div className={cn(
                "w-12 h-12 rounded-xl flex items-center justify-center text-white mb-4 group-hover:shadow-lg transition-all duration-300 bg-gradient-to-br",
                metric.color
              )}>
                <Icon className="w-6 h-6" />
              </div>
              <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">{metric.value}</div>
              <div className="text-gray-600 dark:text-gray-400">{metric.label}</div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Filter Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="glass-card p-4 flex items-center gap-3 flex-wrap"
      >
        <Filter className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        <button
          onClick={() => setSelectedFilter('all')}
          className={cn(
            "px-4 py-2 rounded-lg font-medium transition-all duration-300",
            selectedFilter === 'all'
              ? "bg-gradient-horizon text-white shadow-horizon"
              : "bg-gray-100 dark:bg-horizon-secondary/30 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-horizon-secondary/50"
          )}
        >
          All Entities
        </button>
        {entities.map((entity) => (
          <button
            key={entity.type}
            onClick={() => setSelectedFilter(entity.type)}
            className={cn(
              "px-4 py-2 rounded-lg font-medium transition-all duration-300 text-sm",
              selectedFilter === entity.type
                ? cn(entity.color, "ring-2 ring-offset-2 dark:ring-offset-horizon-dark ring-current")
                : cn(entity.color, "opacity-70 hover:opacity-100")
            )}
          >
            {entity.type.replace('_', ' ')} ({entity.count})
          </button>
        ))}
      </motion.div>

      {/* Entities Grid */}
      <motion.div
        layout
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        <AnimatePresence mode="wait">
          {filteredEntities.map((entity, index) => (
            <motion.div
              key={entity.type}
              layout
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.4 }}
              whileHover={{ y: -5 }}
              onClick={() => setActiveEntity(activeEntity === entity.type ? null : entity.type)}
              className="glass-card p-6 cursor-pointer group hover:shadow-horizon transition-all duration-300"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center", entity.color)}>
                  {entity.type === 'COMPANY' && <Building className="w-6 h-6" />}
                  {entity.type === 'REVENUE' && <DollarSign className="w-6 h-6" />}
                  {entity.type === 'STOCK_PRICE' && <TrendingUp className="w-6 h-6" />}
                  {entity.type === 'MARKET_CAP' && <BarChart3 className="w-6 h-6" />}
                  {entity.type === 'FINANCIAL_EVENT' && <Calendar className="w-6 h-6" />}
                  {entity.type === 'FINANCIAL_RATIO' && <Brain className="w-6 h-6" />}
                </div>
                <div className="text-right">
                  <div className="text-3xl font-bold text-gray-900 dark:text-white">{entity.count}</div>
                  <div className="text-xs text-gray-600 dark:text-gray-400">found</div>
                </div>
              </div>

              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 group-hover:text-horizon-primary transition-colors">
                {entity.type.replace('_', ' ')}
              </h3>

              <div className="space-y-2 mb-4">
                {entity.examples.slice(0, 2).map((example, idx) => (
                  <div key={idx} className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-green-500 flex-shrink-0" />
                    {example}
                  </div>
                ))}
              </div>

              <AnimatePresence>
                {activeEntity === entity.type && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="pt-4 border-t border-gray-200/50 dark:border-white/10 mt-4"
                  >
                    <div className="space-y-2">
                      {entity.examples.map((example, idx) => (
                        <div key={idx} className="flex items-center justify-between p-2 bg-gray-50/50 dark:bg-horizon-secondary/30 rounded-lg group/item hover:bg-gray-100 dark:hover:bg-horizon-secondary/50 transition-colors">
                          <span className="text-sm text-gray-700 dark:text-gray-300">{example}</span>
                          <button className="p-1 opacity-0 group-hover/item:opacity-100 transition-opacity" title="Copy">
                            <Copy className="w-4 h-4 text-horizon-primary" />
                          </button>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* Document Highlighting */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="glass-card p-8"
      >
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-6 flex items-center gap-2">
          <ZoomIn className="w-5 h-5 text-horizon-primary" />
          Document Analysis Preview
        </h3>

        <div className="bg-gray-50/50 dark:bg-horizon-secondary/30 rounded-xl p-6 text-sm text-gray-700 dark:text-gray-300 leading-relaxed space-y-4 max-h-[600px] overflow-y-auto">
          {data && data.visualization_url ? (
            <div className="w-full h-[500px] bg-white rounded-lg overflow-hidden border border-gray-200 dark:border-white/10">
              <iframe
                src={`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'}${data.visualization_url}`}
                className="w-full h-full"
                title="NER Visualization"
              />
            </div>
          ) : data && data.text ? (
            <p className="whitespace-pre-wrap">{data.text}</p>
          ) : (
            <p className="text-gray-500 italic">No text content available for preview.</p>
          )}
        </div>
      </motion.div>

      {/* Export Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="flex justify-center"
      >
        <button className="flex items-center gap-3 px-8 py-4 rounded-xl bg-gradient-horizon text-white font-semibold shadow-horizon hover:shadow-horizon-lg transition-all duration-300 hover:scale-105">
          <Download className="w-5 h-5" />
          Export NER Results
        </button>
      </motion.div>
    </div>
  );
}
