import React from 'react';
import { View, Text } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { useTheme } from '../context/ThemeContext';

interface Props {
  progress: number; // 0–1
  size?: number;
  strokeWidth?: number;
  color?: string;
  label?: string;
  sublabel?: string;
}

export default function ProgressRing({
  progress,
  size = 200,
  strokeWidth = 12,
  color = '#2A7FFF',
  label,
  sublabel,
}: Props) {
  const { isDarkMode } = useTheme();
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - Math.min(progress, 1));

  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={size} height={size} style={{ position: 'absolute' }}>
        {/* Background circle */}
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={isDarkMode ? '#334155' : '#E2E8F0'}
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <Circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          rotation="-90"
          origin={`${size / 2}, ${size / 2}`}
        />
      </Svg>
      <View style={{ alignItems: 'center' }}>
        {label && (
          <Text style={{ fontSize: 32, fontWeight: '800', color: isDarkMode ? '#FFFFFF' : '#1E293B' }}>
            {label}
          </Text>
        )}
        {sublabel && (
          <Text style={{ fontSize: 13, color: '#94A3B8', marginTop: 2 }}>
            {sublabel}
          </Text>
        )}
      </View>
    </View>
  );
}
