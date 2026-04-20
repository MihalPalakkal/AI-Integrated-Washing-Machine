import React from 'react';
import { View, Text } from 'react-native';
import { WashCycle } from '../types/Cycle';
import { Thermometer, RotateCw, Clock, FlaskConical, Droplets, Waves, Timer, RefreshCw } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';

interface Props {
  cycle: WashCycle;
  title?: string;
}

export default function CycleCard({ cycle, title = 'Recommended Cycle' }: Props) {
  const { isDarkMode } = useTheme();
  return (
    <View style={tw`${isDarkMode ? 'bg-dark-card border border-dark-border' : 'bg-white shadow-sm'} rounded-2xl p-5`}>
      <View style={tw`flex-row items-center justify-between mb-4`}>
        <Text style={tw`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>{title}</Text>
        <View style={tw`bg-secondary/15 px-3 py-1 rounded-full`}>
          <Text style={tw`text-secondary text-xs font-bold`}>AI PREDICTED</Text>
        </View>
      </View>

      <View style={tw`gap-3`}>
        <ParamRow
          icon={<Thermometer size={18} color="#FF6B6B" />}
          label="Temperature"
          value={`${cycle.temperature}°C`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<RotateCw size={18} color="#6366F1" />}
          label="Spin Time"
                    value={`${cycle.spinTime} min`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<Clock size={18} color="#2A7FFF" />}
          label="Total Duration"
          value={`${cycle.duration} min`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<Timer size={18} color="#D97706" />}
          label="Soak Time"
          value={`${cycle.soakTime} min`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<FlaskConical size={18} color="#00C9A7" />}
          label="Detergent"
          value={`${cycle.detergent} ml`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<Droplets size={18} color="#3B82F6" />}
          label="Water Usage"
          value={`${cycle.water} L`}
          isDarkMode={isDarkMode}
        />
        <ParamRow
          icon={<RefreshCw size={18} color="#8B5CF6" />}
          label="Wash Cycles"
          value={`${cycle.washCycles}`}
          isDarkMode={isDarkMode}
        />
        {cycle.agitationPattern && (
          <ParamRow
            icon={<Waves size={18} color="#EC4899" />}
            label="Mechanical Action"
            value={cycle.agitationPattern}
            isDarkMode={isDarkMode}
          />
        )}
      </View>
    </View>
  );
}

function ParamRow({ icon, label, value, isDarkMode }: { icon: React.ReactNode; label: string; value: string; isDarkMode: boolean }) {
  return (
    <View style={tw`flex-row items-center justify-between py-2 border-b ${isDarkMode ? 'border-dark-border' : 'border-gray-50'}`}>
      <View style={tw`flex-row items-center gap-3`}>
        <View style={tw`${isDarkMode ? 'bg-dark-border' : 'bg-gray-50'} p-2 rounded-xl`}>{icon}</View>
        <Text style={tw`text-gray-500 text-sm`}>{label}</Text>
      </View>
      <Text style={tw`font-bold ${isDarkMode ? 'text-slate-200' : 'text-gray-800'}`}>{value}</Text>
    </View>
  );
}
