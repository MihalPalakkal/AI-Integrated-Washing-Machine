import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, RefreshControl, TouchableOpacity } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Clock, Droplets, FlaskConical, ChevronDown, ChevronUp, Activity } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { onWashHistoryChange } from '../services/firebase';
import { WashHistoryEntry } from '../types/Cycle';
import { useTheme } from '../context/ThemeContext';

export default function HistoryScreen() {
  const insets = useSafeAreaInsets();
  const { isDarkMode } = useTheme();
  const [history, setHistory] = useState<WashHistoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    const unsub = onWashHistoryChange((data) => {
      setHistory(data);
      setLoading(false);
    });
    return () => unsub();
  }, []);

  const renderItem = ({ item }: { item: WashHistoryEntry }) => {
    const isExpanded = expandedId === item.id;
    const date = new Date(item.date);
    const dateStr = date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
    const timeStr = date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });

    return (
      <TouchableOpacity
        onPress={() => setExpandedId(isExpanded ? null : item.id)}
        activeOpacity={0.7}
        style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-4 mb-3`}
      >
        {/* Header Row */}
        <View style={tw`flex-row justify-between items-center`}>
          <View style={tw`flex-1`}>
            <View style={tw`flex-row items-center gap-2`}>
              <View style={tw`p-1.5 rounded-lg bg-blue-100`}>
                <Activity size={12} color="#2A7FFF" />
              </View>
              <Text style={tw`font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'} text-base`}>{item.cycleUsed}</Text>
            </View>
            <Text style={tw`text-gray-400 text-xs mt-1 ml-7`}>{dateStr} at {timeStr}</Text>
          </View>
          <View style={tw`flex-row items-center gap-2`}>
            <View style={tw`bg-primary/10 px-3 py-1 rounded-full`}>
              <Text style={tw`text-primary text-xs font-bold`}>{item.duration}m</Text>
            </View>
            {isExpanded ? (
              <ChevronUp size={18} color="#94A3B8" />
            ) : (
              <ChevronDown size={18} color="#94A3B8" />
            )}
          </View>
        </View>

        {/* Expanded Details */}
        {isExpanded && (
          <View style={tw`mt-4 pt-4 border-t ${isDarkMode ? 'border-dark-border' : 'border-gray-100'}`}>
            <DetailRow
              icon={<View style={tw`w-2 h-2 rounded-full bg-primary`} />}
              label="Fabric Type"
              value={item.fabricDetected}
              isDarkMode={isDarkMode}
            />
            <DetailRow
              icon={<Droplets size={14} color="#2A7FFF" />}
              label="Water Used"
              value={`${item.waterConsumed} L`}
              isDarkMode={isDarkMode}
            />
            <DetailRow
              icon={<FlaskConical size={14} color="#00C9A7" />}
              label="Detergent Used"
              value={`${item.detergentConsumed} ml`}
              isDarkMode={isDarkMode}
            />
            <DetailRow
              icon={<Clock size={14} color="#8B5CF6" />}
              label="Actual Time"
              value={`${item.duration} min`}
              isDarkMode={isDarkMode}
            />
          </View>
        )}
      </TouchableOpacity>
    );
  };

  return (
    <View style={[tw`flex-1 ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`, { paddingTop: insets.top }]}>
      <View style={tw`p-5 pb-2`}>
        <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Wash History</Text>
        <Text style={tw`text-gray-400 text-xs mt-0.5`}>Previous wash sessions</Text>
      </View>
      <FlatList
        data={history}
        renderItem={renderItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{ padding: 20, paddingBottom: 100 }}
        refreshControl={<RefreshControl refreshing={loading} onRefresh={() => {}} />}
        ListEmptyComponent={
          <View style={tw`items-center py-20`}>
            <Text style={tw`text-gray-400`}>No history available yet.</Text>
          </View>
        }
      />
    </View>
  );
}

function DetailRow({ icon, label, value, isDarkMode }: { icon: React.ReactNode; label: string; value: string; isDarkMode: boolean }) {
  return (
    <View style={tw`flex-row justify-between items-center py-2`}>
      <View style={tw`flex-row items-center gap-2`}>
        {icon}
        <Text style={tw`text-gray-500 text-sm`}>{label}</Text>
      </View>
      <Text style={tw`font-bold ${isDarkMode ? 'text-slate-200' : 'text-gray-800'} text-sm`}>{value}</Text>
    </View>
  );
}
