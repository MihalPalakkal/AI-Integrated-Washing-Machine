import React from 'react';
import { View, Text } from 'react-native';
import { Check } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { MachineStatus } from '../types/Machine';
import { useTheme } from '../context/ThemeContext';

const WASH_STEPS: { key: string; label: string }[] = [
  { key: 'DATA_FETCH', label: 'Params' },
  { key: 'DRUM', label: 'Balance' },
  { key: 'DETERGENT', label: 'Detergent' },
  { key: 'WATER_FILL', label: 'Filling' },
  { key: 'SOAK', label: 'Soaking' },
  { key: 'SPIN', label: 'Spinning' },
  { key: 'completed', label: 'Done' },
];

interface Props {
  currentStatus: string;
}

export default function StepIndicator({ currentStatus }: Props) {
  const { isDarkMode } = useTheme();
  const currentIndex = WASH_STEPS.findIndex((s) => s.key === currentStatus);

  return (
    <View style={tw`flex-row items-center justify-between px-2`}>
      {WASH_STEPS.map((step, index) => {
        const isCompleted = index < currentIndex;
        const isActive = index === currentIndex;
        const isPending = index > currentIndex;

        return (
          <View key={step.key} style={tw`items-center flex-1`}>
            {/* Dot / Check */}
            <View
              style={[
                tw`w-8 h-8 rounded-full items-center justify-center mb-1.5`,
                isCompleted && { backgroundColor: '#00C9A7' },
                isActive && { backgroundColor: '#2A7FFF' },
                isPending && (isDarkMode ? tw`bg-dark-border` : tw`bg-gray-200`),
              ]}
            >
              {isCompleted ? (
                <Check size={14} color="white" />
              ) : isActive ? (
                <View style={tw`w-2.5 h-2.5 bg-white rounded-full`} />
              ) : (
                <View style={[tw`w-2 h-2 rounded-full`, { backgroundColor: isDarkMode ? '#475569' : '#94A3B8' }]} />
              )}
            </View>
            <Text
              style={[
                tw`text-xs text-center`,
                isActive && tw`text-primary font-bold`,
                isCompleted && tw`text-secondary font-medium`,
                isPending && (isDarkMode ? tw`text-dark-muted` : tw`text-gray-400`),
              ]}
              numberOfLines={1}
            >
              {step.label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}
