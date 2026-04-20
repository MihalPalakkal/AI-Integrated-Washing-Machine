import React from 'react';
import { View, Text } from 'react-native';
import { Check, Activity } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';

interface LogEntry {
  event: string;
  message: string;
  time_ms: number;
}

interface Props {
  logs: Record<string, LogEntry>;
  machineStatus: string;
}

export default function VerticalStepIndicator({ logs, machineStatus }: Props) {
  const { isDarkMode } = useTheme();
  
  // Transform, filter out WiFi connection, and sort logs by time
  const sortedLogs = Object.values(logs)
    .filter(log => log.event !== 'SYSTEM')
    .sort((a, b) => a.time_ms - b.time_ms);

  if (sortedLogs.length === 0 || machineStatus === 'idle') {
    return (
      <View style={tw`items-center py-10`}>
        <Text style={tw`text-gray-400 italic`}>
          {machineStatus === 'idle' ? 'Ready to start wash cycle' : 'Waiting for hardware logs...'}
        </Text>
      </View>
    );
  }

  return (
    <View style={tw`w-full pt-1`}>
      {sortedLogs.map((log, index) => {
        const isLatest = index === sortedLogs.length - 1; 
        
        return (
          <View 
            key={`${log.event}-${log.time_ms}`} 
            style={[
              tw`flex-row items-center p-4 mb-3 rounded-2xl shadow-sm`,
              isLatest ? (isDarkMode ? tw`bg-blue-900/30` : tw`bg-blue-50`) : (isDarkMode ? tw`bg-gray-800/50` : tw`bg-gray-50/80`),
              isLatest && tw`border border-blue-500/30`,
            ]}
          >
            {/* Number Badge */}
            <View 
              style={[
                tw`w-9 h-9 rounded-full items-center justify-center mr-4 shadow-sm`,
                isLatest ? tw`bg-blue-500` : (isDarkMode ? tw`bg-gray-700` : tw`bg-white`)
              ]}
            >
              <Text style={tw`text-[15px] font-black ${isLatest ? 'text-white' : (isDarkMode ? 'text-gray-300' : 'text-gray-700')}`}>
                {index + 1}
              </Text>
            </View>

            {/* Message Content */}
            <View style={tw`flex-1`}>
              <Text 
                style={[
                  tw`text-sm font-bold tracking-wide`, 
                  isLatest ? (isDarkMode ? tw`text-blue-400` : tw`text-blue-700`) : (isDarkMode ? tw`text-gray-300` : tw`text-gray-700`)
                ]}
              >
                {log.message} {log.event === 'DRAIN_END' ? '✨' : ''}
              </Text>
              
              {isLatest && (
                 <Text style={tw`text-[10px] font-black text-blue-500/70 uppercase tracking-widest mt-1`}>Currently Active</Text>
              )}
            </View>
            
            {/* Right side Icon */}
            {isLatest ? (
              <View style={tw`bg-blue-500/10 p-2 rounded-full ml-3`}>
                <Activity size={18} color="#3B82F6" />
              </View>
            ) : (
              <View style={tw`${isDarkMode ? 'bg-gray-700/50' : 'bg-emerald-50'} p-1.5 rounded-full ml-3`}>
                <Check size={16} color="#10B981" strokeWidth={3} />
              </View>
            )}
          </View>
        );
      })}
    </View>
  );
}
