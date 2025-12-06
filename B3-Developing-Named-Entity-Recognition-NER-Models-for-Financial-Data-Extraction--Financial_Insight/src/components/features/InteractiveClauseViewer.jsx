import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    ChevronLeft, Chrome, ChevronRight, Pause, Play, Info,
    Search, Filter, ArrowDown, ExternalLink, Shield, AlertTriangle, CheckCircle, FileText
} from 'lucide-react';
import { cn } from '../../lib/utils';

export default function InteractiveClauseViewer({ text, clauses }) {
    const [activeClauseIndex, setActiveClauseIndex] = useState(-1);
    const [isPlaying, setIsPlaying] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");
    const [filterType, setFilterType] = useState("all");

    const scrollRef = useRef(null);
    const sidebarRef = useRef(null);

    // Memoize sorted and filtered clauses
    const sortedClauses = React.useMemo(() => {
        let processed = [...clauses]
            .filter(c => c.text && text.includes(c.text))
            .sort((a, b) => text.indexOf(a.text) - text.indexOf(b.text));

        if (searchTerm) {
            processed = processed.filter(c =>
                c.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
                c.type.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        if (filterType !== "all") {
            processed = processed.filter(c => c.type === filterType);
        }

        return processed;
    }, [clauses, text, searchTerm, filterType]);

    // Unique clause types for filter
    const clauseTypes = React.useMemo(() => {
        return ["all", ...new Set(clauses.map(c => c.type))];
    }, [clauses]);

    // Auto-play logic
    useEffect(() => {
        let interval;
        if (isPlaying && sortedClauses.length > 0) {
            interval = setInterval(() => {
                setActiveClauseIndex(prev => {
                    // Find current index in the filtered list
                    const currentFilteredIndex = sortedClauses.findIndex(c => c === clauses[prev]);
                    // Note: This logic is slighty complex because indices shift with filters. 
                    // Simplified: Cycle through 0 to length-1 of sortedClauses

                    if (prev >= sortedClauses.length - 1) {
                        setIsPlaying(false);
                        return prev;
                    }
                    return prev + 1;
                });
            }, 2500);
        }
        return () => clearInterval(interval);
    }, [isPlaying, sortedClauses.length]);

    // Scroll to active clause in Document View
    useEffect(() => {
        if (activeClauseIndex >= 0 && activeClauseIndex < sortedClauses.length && scrollRef.current) {
            const activeElement = scrollRef.current.querySelector(`[data-index="${activeClauseIndex}"]`);
            if (activeElement) {
                activeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [activeClauseIndex, sortedClauses]);

    // Scroll to active clause card in Sidebar
    useEffect(() => {
        if (activeClauseIndex >= 0 && activeClauseIndex < sortedClauses.length && sidebarRef.current) {
            const activeCard = sidebarRef.current.querySelector(`[data-card-index="${activeClauseIndex}"]`);
            if (activeCard) {
                activeCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }
    }, [activeClauseIndex, sortedClauses]);


    const getHighlightColor = (type, isActive) => {
        const typeLower = type ? type.toLowerCase() : 'default';

        if (typeLower.includes('liability'))
            return isActive
                ? 'bg-red-200 dark:bg-red-500/40 text-red-900 dark:text-white ring-2 ring-red-500'
                : 'bg-red-100 dark:bg-red-500/20 text-red-800 dark:text-red-200 border-b-2 border-red-300 dark:border-red-700/50';

        if (typeLower.includes('payment') || typeLower.includes('money'))
            return isActive
                ? 'bg-green-200 dark:bg-green-500/40 text-green-900 dark:text-white ring-2 ring-green-500'
                : 'bg-green-100 dark:bg-green-500/20 text-green-800 dark:text-green-200 border-b-2 border-green-300 dark:border-green-700/50';

        if (typeLower.includes('confidential'))
            return isActive
                ? 'bg-purple-200 dark:bg-purple-500/40 text-purple-900 dark:text-white ring-2 ring-purple-500'
                : 'bg-purple-100 dark:bg-purple-500/20 text-purple-800 dark:text-purple-200 border-b-2 border-purple-300 dark:border-purple-700/50';

        if (typeLower.includes('termination'))
            return isActive
                ? 'bg-orange-200 dark:bg-orange-500/40 text-orange-900 dark:text-white ring-2 ring-orange-500'
                : 'bg-orange-100 dark:bg-orange-500/20 text-orange-800 dark:text-orange-200 border-b-2 border-orange-300 dark:border-orange-700/50';

        // Default
        return isActive
            ? 'bg-blue-200 dark:bg-blue-500/40 text-blue-900 dark:text-white ring-2 ring-blue-500'
            : 'bg-blue-100 dark:bg-blue-500/20 text-blue-800 dark:text-blue-200 border-b-2 border-blue-300 dark:border-blue-700/50';
    };

    const renderHighlightedText = () => {
        if (!text) return <p className="text-gray-500 italic">No text content available.</p>;

        let lastIndex = 0;
        const elements = [];

        // Map sortedClauses to highlighting Logic
        // Need to handle overlaps? For MVP assume no huge overlaps or simple rendering

        sortedClauses.forEach((clause, idx) => {
            const startIndex = text.indexOf(clause.text, lastIndex);
            // Note: text.indexOf from lastIndex is safer for duplicates but assumes order
            // Ideally we would strictly order by found index first.

            const absoluteIndex = text.indexOf(clause.text); // Simplified for this demo (ignores dupes after first)
            if (absoluteIndex === -1) return;

            // If we jumped backwards (because sortedClauses was possibly re-sorted by filter etc but indexOf finds first instance), 
            // we might have rendering artifacts. 
            // FIX: We should map all occurrences first, THEN sort by position. 
            // For this UI, we will rely on strict order passed in to work best.

            const displayStart = startIndex !== -1 ? startIndex : text.indexOf(clause.text); // Fallback
            if (displayStart < lastIndex) return; // Skip overlapping/backwards for now to prevent corruption

            // Text before highlight
            if (displayStart > lastIndex) {
                elements.push(<span key={`text-${idx}`}>{text.substring(lastIndex, displayStart)}</span>);
            }

            // Highlighted text
            elements.push(
                <motion.span
                    key={`highlight-${idx}`}
                    layoutId={`highlight-${idx}`}
                    data-index={idx}
                    onClick={() => setActiveClauseIndex(idx)}
                    className={cn(
                        "rounded px-1 transition-all duration-200 cursor-pointer mx-0.5 inline-block",
                        getHighlightColor(clause.type, idx === activeClauseIndex)
                    )}
                >
                    {text.substring(displayStart, displayStart + clause.text.length)}
                </motion.span>
            );

            lastIndex = displayStart + clause.text.length;
        });

        if (lastIndex < text.length) {
            elements.push(<span key="text-end">{text.substring(lastIndex)}</span>);
        }

        return elements;
    };

    return (
        <div className="flex flex-col lg:flex-row gap-6 h-[800px] lg:h-[700px]">
            {/* LEFT COLUMN: DOCUMENT VIEWER */}
            <div className="flex-1 flex flex-col bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
                {/* Toolbar */}
                <div className="flex items-center justify-between p-4 border-b border-gray-100 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-800/50 backdrop-blur-sm">
                    <div className="flex items-center gap-2">
                        <FileText className="w-5 h-5 text-gray-500" />
                        <span className="font-semibold text-gray-700 dark:text-gray-200 text-sm">Document Text</span>
                    </div>

                    <div className="flex items-center gap-2 bg-white dark:bg-gray-900 rounded-lg p-1 border border-gray-200 dark:border-gray-700">
                        <button
                            onClick={() => setActiveClauseIndex(Math.max(0, activeClauseIndex - 1))}
                            disabled={activeClauseIndex <= 0}
                            className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md transition-colors disabled:opacity-30"
                        >
                            <ChevronLeft className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                        </button>
                        <span className="text-xs font-mono w-16 text-center text-gray-500">
                            {sortedClauses.length > 0 ? activeClauseIndex + 1 : 0} / {sortedClauses.length}
                        </span>
                        <button
                            onClick={() => setActiveClauseIndex(Math.min(sortedClauses.length - 1, activeClauseIndex + 1))}
                            disabled={activeClauseIndex >= sortedClauses.length - 1}
                            className="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-md transition-colors disabled:opacity-30"
                        >
                            <ChevronRight className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                        </button>
                        <div className="w-px h-4 bg-gray-200 dark:bg-gray-700 mx-1"></div>
                        <button
                            onClick={() => setIsPlaying(!isPlaying)}
                            className={cn(
                                "p-1.5 rounded-md transition-colors",
                                isPlaying ? "bg-red-50 text-red-500" : "hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
                            )}
                            title={isPlaying ? "Pause Auto-play" : "Start Auto-play"}
                        >
                            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                        </button>
                    </div>
                </div>

                {/* Content of Text */}
                <div
                    ref={scrollRef}
                    className="flex-1 overflow-y-auto p-8 whitespace-pre-wrap font-serif text-lg leading-loose text-gray-800 dark:text-gray-300 selection:bg-horizon-primary/20"
                >
                    {renderHighlightedText()}
                </div>
            </div>

            {/* RIGHT COLUMN: SIDEBAR */}
            <div className="w-full lg:w-96 flex flex-col bg-gray-50 dark:bg-gray-800/50 rounded-2xl border border-gray-200 dark:border-gray-700 overflow-hidden">
                {/* Header & Filter */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-700 space-y-4">
                    <div className="flex items-center justify-between">
                        <h3 className="font-bold text-gray-900 dark:text-white">Detected Clauses</h3>
                        <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded-full text-gray-600 dark:text-gray-300">
                            {sortedClauses.length}
                        </span>
                    </div>

                    <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                            type="text"
                            placeholder="Search clauses..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-9 pr-4 py-2 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:ring-2 focus:ring-horizon-primary focus:border-transparent outline-none"
                        />
                    </div>

                    <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                        {clauseTypes.map(type => (
                            <button
                                key={type}
                                onClick={() => setFilterType(type)}
                                className={cn(
                                    "whitespace-nowrap px-3 py-1 rounded-full text-xs font-medium transition-all border",
                                    filterType === type
                                        ? "bg-gray-900 dark:bg-white text-white dark:text-gray-900 border-transparent shadow-md"
                                        : "bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-gray-700 hover:border-gray-400"
                                )}
                            >
                                {type === "all" ? "All Types" : type}
                            </button>
                        ))}
                    </div>
                </div>

                {/* List */}
                <div ref={sidebarRef} className="flex-1 overflow-y-auto p-4 space-y-3">
                    {sortedClauses.map((clause, idx) => (
                        <motion.div
                            key={idx}
                            data-card-index={idx}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            onClick={() => setActiveClauseIndex(idx)}
                            className={cn(
                                "group p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden",
                                activeClauseIndex === idx
                                    ? "bg-white dark:bg-gray-900 border-horizon-primary shadow-lg ring-1 ring-horizon-primary/50"
                                    : "bg-white dark:bg-gray-900/50 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:shadow-md"
                            )}
                        >
                            <div className="flex justify-between items-start mb-2 relative z-10">
                                <span className={cn(
                                    "text-xs font-bold uppercase tracking-wider px-2 py-1 rounded-md",
                                    activeClauseIndex === idx
                                        ? "bg-horizon-primary text-white"
                                        : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 group-hover:bg-gray-200 dark:group-hover:bg-gray-700"
                                )}>
                                    {clause.type}
                                </span>
                                {activeClauseIndex === idx && (
                                    <ArrowDown className="w-4 h-4 text-horizon-primary -rotate-90" />
                                )}
                            </div>

                            <p className="text-sm text-gray-800 dark:text-gray-300 line-clamp-2 mb-2 font-medium relative z-10">
                                "{clause.text}"
                            </p>

                            {clause.attributes && Object.keys(clause.attributes).length > 0 && (
                                <div className="flex flex-wrap gap-2 relative z-10">
                                    {Object.entries(clause.attributes).slice(0, 2).map(([k, v]) => (
                                        k !== 'confidence' && (
                                            <div key={k} className="flex items-center gap-1 text-[10px] text-gray-500 bg-gray-50 dark:bg-gray-800 px-2 py-0.5 rounded border border-gray-100 dark:border-gray-700">
                                                <span className="opacity-70">{k}:</span>
                                                <span className="font-semibold">{String(v)}</span>
                                            </div>
                                        )
                                    ))}
                                </div>
                            )}

                            {/* Active Indicator Bar */}
                            {activeClauseIndex === idx && (
                                <div className="absolute left-0 top-0 bottom-0 w-1 bg-horizon-primary" />
                            )}
                        </motion.div>
                    ))}

                    {sortedClauses.length === 0 && (
                        <div className="text-center py-10 text-gray-400">
                            <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                            <p>No clauses match your search.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
