import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
    PieChart, Pie, Cell,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { motion } from 'framer-motion';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export default function AnalysisCharts({ data }) {
    if (!data) return null;

    // Prepare data for Entity Distribution (NER)
    const entityData = React.useMemo(() => {
        if (!data.ner || !data.ner.entities) return [];
        const counts = {};
        data.ner.entities.forEach(e => {
            counts[e.label] = (counts[e.label] || 0) + 1;
        });

        // Sort by value (descending) and take top 10
        return Object.keys(counts)
            .map(key => ({
                name: key,
                value: counts[key]
            }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 10);
    }, [data.ner]);

    // Prepare data for Sentiment Distribution
    const sentimentData = React.useMemo(() => {
        if (!data.finbert) return [];

        // Use statistics if available (new backend format)
        if (data.finbert.statistics && data.finbert.statistics.sentiment_distribution) {
            const dist = data.finbert.statistics.sentiment_distribution;
            return [
                { name: 'Positive', value: dist.positive || 0 },
                { name: 'Negative', value: dist.negative || 0 },
                { name: 'Neutral', value: dist.neutral || 0 },
            ];
        }

        // Fallback
        return [
            { name: 'Positive', value: 0 },
            { name: 'Negative', value: 0 },
            { name: 'Neutral', value: 0 },
        ];
    }, [data.finbert]);

    // Prepare data for Contract Risk Profile
    const riskData = React.useMemo(() => {
        if (!data.langextract || !data.langextract.clauses) return [];

        const riskLevels = { High: 0, Medium: 0, Low: 0 };

        data.langextract.clauses.forEach(c => {
            // Handle both new backend format (extraction_class) and old format (type/class)
            const type = (c.extraction_class || c.type || c.class || 'General Clause').toLowerCase();

            // Logic must match ClauseExtraction.jsx for consistency
            if (type.includes('liability') || type.includes('termination') || type.includes('indemnity') || type.includes('payment') || type.includes('dispute')) {
                riskLevels.High++;
            } else if (type.includes('confidentiality') || type.includes('warranty') || type.includes('governing') || type.includes('interest')) {
                riskLevels.Medium++;
            } else {
                riskLevels.Low++;
            }
        });

        const total = Object.values(riskLevels).reduce((a, b) => a + b, 0);
        if (total === 0) return [];

        return [
            { name: 'High Risk', value: riskLevels.High, color: '#ef4444' },   // Red-500
            { name: 'Medium Risk', value: riskLevels.Medium, color: '#f59e0b' }, // Amber-500
            { name: 'Low Risk', value: riskLevels.Low, color: '#10b981' }      // Emerald-500
        ].filter(item => item.value > 0);
    }, [data.langextract]);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            {/* Entity Distribution Chart */}
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="glass-card p-6"
            >
                <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">Entity Distribution</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={entityData}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, percent }) => percent > 0.1 ? `${name} ${(percent * 100).toFixed(0)}%` : ''}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                            >
                                {entityData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.9)', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }} />
                            <Legend layout="vertical" align="right" verticalAlign="middle" wrapperStyle={{ fontSize: '12px' }} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* Sentiment Analysis Chart */}
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="glass-card p-6"
            >
                <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">Sentiment Distribution (Sentence Count)</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={sentimentData}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                            <XAxis dataKey="name" stroke="#888" />
                            <YAxis stroke="#888" />
                            <Tooltip contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.8)', borderRadius: '8px' }} />
                            <Bar dataKey="value" fill="#82ca9d" radius={[4, 4, 0, 0]}>
                                {sentimentData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.name === 'Positive' ? '#00C49F' : entry.name === 'Negative' ? '#FF8042' : '#FFBB28'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* Contract Risk Profile */}
            {riskData.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    className="glass-card p-6 md:col-span-2"
                >
                    <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">Contract Risk Profile</h3>
                    <div className="flex flex-col md:flex-row items-center justify-center gap-8">
                        <div className="h-64 w-full md:w-1/2">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie
                                        data={riskData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={80}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {riskData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.9)', borderRadius: '8px' }} />
                                    <Legend verticalAlign="middle" align="right" layout="vertical" />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="w-full md:w-1/2 space-y-4">
                            <p className="text-sm text-gray-600 dark:text-gray-300">
                                This analysis categorizes extracted clauses based on their potential legal and financial impact.
                            </p>
                            <div className="grid grid-cols-1 gap-3">
                                {riskData.map((item) => (
                                    <div key={item.name} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                                        <div className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                                            <span className="font-medium text-gray-700 dark:text-gray-200">{item.name}</span>
                                        </div>
                                        <span className="font-bold text-gray-900 dark:text-white">{item.value} Clauses</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
