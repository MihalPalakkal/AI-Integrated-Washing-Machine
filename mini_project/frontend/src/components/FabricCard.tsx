import React from 'react';
import { View, Text } from 'react-native';
import { FabricDetection } from '../types/Fabric';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';

interface Props {
  fabric: FabricDetection;
  isDominant?: boolean;
}

// Dynamic color based on fiber category
function getCategoryColor(category: string): string {
  switch (category.toLowerCase()) {
    case 'natural': return '#059669';
    case 'synthetic': return '#6366F1';
    case 'semi-synthetic': return '#D97706';
    default: return '#64748B';
  }
}

// Dirt level colors
function getDirtColor(level: number): string {
  if (level <= 1) return '#00C9A7';
  if (level <= 2) return '#34D399';
  if (level <= 3) return '#F59E0B';
  if (level <= 4) return '#F97316';
  return '#EF4444';
}

function getDirtLabel(level: number): string {
  const labels = ['', 'Clean', 'Light Soil', 'Moderate', 'Heavy', 'Emergency'];
  return labels[level] || 'Unknown';
}

export default function FabricCard({ fabric, isDominant }: Props) {
  const { isDarkMode } = useTheme();
  const categoryColor = getCategoryColor(fabric.fiberCategory);
  const percent = Math.round(fabric.confidence * 100);
  const dirtColor = getDirtColor(fabric.dirtLevel);

  return (
    <View
      style={[
        tw`${isDarkMode ? 'bg-dark-card border-dark-border' : 'bg-white'} rounded-2xl p-4 mb-3`,
        isDominant && { borderWidth: 2, borderColor: categoryColor },
      ]}
    >
      {/* Header Row */}
      <View style={tw`flex-row justify-between items-center mb-2`}>
        <View style={tw`flex-row items-center gap-3 flex-1`}>
          <View style={[tw`w-3 h-3 rounded-full`, { backgroundColor: categoryColor }]} />
          <Text style={tw`font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'} text-base`} numberOfLines={1}>
            {fabric.name}
          </Text>
          {isDominant && (
            <View style={[tw`px-2 py-0.5 rounded-full`, { backgroundColor: categoryColor + '20' }]}>
              <Text style={[tw`text-xs font-bold`, { color: categoryColor }]}>DOMINANT</Text>
            </View>
          )}
        </View>
        <Text style={[tw`font-bold text-lg`, { color: categoryColor }]}>{percent}%</Text>
      </View>

      {/* Fiber Category Badge */}
      <View style={tw`flex-row gap-2 mb-3`}>
        <View style={[tw`px-3 py-1 rounded-full`, { backgroundColor: categoryColor + '15' }]}>
          <Text style={[tw`text-xs font-bold`, { color: categoryColor }]}>
            {fabric.fiberCategory}
          </Text>
        </View>
        <View style={[tw`px-3 py-1 rounded-full`, { backgroundColor: dirtColor + '15' }]}>
          <Text style={[tw`text-xs font-bold`, { color: dirtColor }]}>
            Dirt: {getDirtLabel(fabric.dirtLevel)} ({fabric.dirtLevel}/5)
          </Text>
        </View>
      </View>

      {/* Confidence Bar */}
      <View style={tw`h-2.5 ${isDarkMode ? 'bg-dark-border' : 'bg-gray-100'} rounded-full overflow-hidden mb-3`}>
        <View
          style={[
            tw`h-full rounded-full`,
            { backgroundColor: categoryColor, width: `${percent}%` },
          ]}
        />
      </View>

      {/* Description */}
      {fabric.description ? (
        <Text style={tw`${isDarkMode ? 'text-dark-muted' : 'text-gray-500'} text-xs leading-4`} numberOfLines={3}>
          {fabric.description}
        </Text>
      ) : null}
    </View>
  );
}
