import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, ActivityIndicator, Image, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Sparkles, Eye, Camera, Image as ImageIcon, ArrowRight, RotateCcw, X, Plus } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import tw from '../lib/tailwind';
import { api } from '../services/api';
import { FabricAnalysisResult } from '../types/Fabric';
import { WashCycle } from '../types/Cycle';
import FabricCard from '../components/FabricCard';
import CycleCard from '../components/CycleCard';
import { useTheme } from '../context/ThemeContext';

type PipelineStage = 'idle' | 'picking' | 'analyzing' | 'predicting' | 'done' | 'error';

export default function FabricAnalysisScreen({ navigation }: any) {
  const insets = useSafeAreaInsets();
  const { isDarkMode } = useTheme();
  const [imageUris, setImageUris] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState<FabricAnalysisResult | null>(null);
  const [cycle, setCycle] = useState<WashCycle | null>(null);
  const [stage, setStage] = useState<PipelineStage>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const pickImage = async (useCamera: boolean) => {
    if (imageUris.length >= 5) {
      Alert.alert('Limit Reached', 'You can upload a maximum of 5 images.');
      return;
    }

    try {
      // Request permissions
      if (useCamera) {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission Required', 'Camera access is needed to take photos.');
          return;
        }
      } else {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
          Alert.alert('Permission Required', 'Gallery access is needed to select photos.');
          return;
        }
      }

      const result = useCamera
        ? await ImagePicker.launchCameraAsync({
            mediaTypes: ['images'],
            quality: 0.8,
          })
        : await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['images'],
            quality: 0.8,
          });

      if (!result.canceled && result.assets.length > 0) {
        setImageUris(prev => [...prev, result.assets[0].uri]);
      }
    } catch (err) {
      console.error('Image picker error:', err);
      Alert.alert('Error', 'Failed to open image picker.');
    }
  };

  const removeImage = (index: number) => {
    setImageUris(prev => prev.filter((_, i) => i !== index));
  };

  const startAnalysis = () => {
    if (imageUris.length === 0) return;
    runPipeline(imageUris);
  };

  const runPipeline = async (uris: string[]) => {
    setStage('analyzing');
    setErrorMsg(null);
    setAnalysis(null);
    setCycle(null);

    try {
      // Step 1: API-1 (Fabric Detection)
      setStage('analyzing');
      const pipelineResult = await api.fullPipeline(uris);

      setAnalysis(pipelineResult.fabricResult);

      // Step 2: API-2 results already in pipeline
      setStage('predicting');
      setCycle(pipelineResult.washCycle);

      // Update recommended cycle name based on API-2 result
      if (pipelineResult.washCycle.agitationPattern) {
        setAnalysis(prev => prev ? {
          ...prev,
          recommendedCycle: `${pipelineResult.washCycle.agitationPattern} Wash — ${pipelineResult.washCycle.temperature}°C`,
        } : prev);
      }

      setStage('done');
    } catch (err: any) {
      console.error('Pipeline error:', err);
      setStage('error');
      setErrorMsg(err.message || 'Analysis failed');
    }
  };

  const reset = () => {
    setImageUris([]);
    setAnalysis(null);
    setCycle(null);
    setStage('idle');
    setErrorMsg(null);
  };

  const isLoading = stage === 'analyzing' || stage === 'predicting';

  return (
    <ScrollView
      style={tw`flex-1 ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}
      contentContainerStyle={{ paddingTop: insets.top, paddingBottom: 100 }}
    >
      <View style={tw`p-5`}>
        {/* Header */}
        <View style={tw`flex-row items-center gap-3 mb-2`}>
          <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-primary/10'} p-2.5 rounded-xl`}>
            <Eye size={22} color="#2A7FFF" />
          </View>
          <View>
            <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Fabric Analysis</Text>
            <Text style={tw`text-gray-400 text-xs`}>AI Vision Detection — API Pipeline</Text>
          </View>
        </View>

        {/* Multi Image Picker UI */}
        {stage === 'idle' && (
          <View style={tw`mt-5`}>
            <Text style={tw`text-sm font-bold ${isDarkMode ? 'text-slate-400' : 'text-gray-600'} mb-3`}>
              Upload 1 to 5 clothing images for accurate prediction
            </Text>
            
            {/* Thumbnail Strip */}
            {imageUris.length > 0 && (
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={tw`flex-row mb-4`}>
                {imageUris.map((uri, index) => (
                  <View key={uri} style={tw`mr-3 relative`}>
                    <Image source={{ uri }} style={tw`w-24 h-24 rounded-2xl`} />
                    <TouchableOpacity 
                      onPress={() => removeImage(index)}
                      style={tw`absolute -top-1 -right-1 bg-alert w-6 h-6 rounded-full items-center justify-center border-2 border-white`}
                    >
                      <X size={14} color="white" />
                    </TouchableOpacity>
                  </View>
                ))}
                {imageUris.length < 5 && (
                  <TouchableOpacity 
                    onPress={() => pickImage(false)}
                    style={tw`w-24 h-24 rounded-2xl border-2 border-dashed ${isDarkMode ? 'border-slate-700 bg-dark-card' : 'border-gray-300 bg-gray-50'} items-center justify-center`}
                  >
                    <Plus size={24} color={isDarkMode ? '#64748B' : '#94A3B8'} />
                  </TouchableOpacity>
                )}
              </ScrollView>
            )}

            <View style={tw`flex-row gap-3`}>
              {imageUris.length === 0 ? (
                <>
                  <TouchableOpacity
                    onPress={() => pickImage(true)}
                    style={tw`flex-1 bg-primary py-4 rounded-2xl flex-row items-center justify-center gap-2`}
                  >
                    <Camera size={20} color="white" />
                    <Text style={tw`text-white font-bold text-base`}>Camera</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => pickImage(false)}
                    style={tw`flex-1 ${isDarkMode ? 'bg-dark-card border-dark-border' : 'bg-white border-primary'} border-2 py-4 rounded-2xl flex-row items-center justify-center gap-2`}
                  >
                    <ImageIcon size={20} color="#2A7FFF" />
                    <Text style={tw`text-primary font-bold text-base`}>Gallery</Text>
                  </TouchableOpacity>
                </>
              ) : (
                <TouchableOpacity
                  onPress={startAnalysis}
                  style={tw`flex-1 bg-primary py-4 rounded-2xl flex-row items-center justify-center gap-2`}
                >
                  <Sparkles size={20} color="white" />
                  <Text style={tw`text-white font-bold text-base`}>Analyze {imageUris.length} {imageUris.length === 1 ? 'Photo' : 'Photos'}</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
        )}

        {/* Selected Images View (After Done) */}
        {stage === 'done' && imageUris.length > 0 && (
          <View style={tw`mt-4`}>
             <ScrollView horizontal showsHorizontalScrollIndicator={false} style={tw`flex-row mb-2`}>
                {imageUris.map((uri) => (
                  <Image key={uri} source={{ uri }} style={tw`w-32 h-32 rounded-2xl mr-3`} />
                ))}
              </ScrollView>
            <TouchableOpacity
              onPress={reset}
              style={tw`mt-3 ${isDarkMode ? 'bg-dark-card' : 'bg-gray-100'} py-3 rounded-2xl flex-row items-center justify-center gap-2`}
            >
              <RotateCcw size={16} color="#64748B" />
              <Text style={tw`${isDarkMode ? 'text-slate-300' : 'text-gray-600'} font-bold text-sm`}>Analyze Another</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Pipeline Progress */}
        {isLoading && (
          <View style={tw`mt-6 items-center py-8 ${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl`}>
            <ActivityIndicator size="large" color="#2A7FFF" />
            <Text style={tw`text-primary font-bold mt-4 text-base`}>
              {stage === 'analyzing' ? '📸 Identifying Fabric...' : '🔧 Predicting Wash Params...'}
            </Text>
            <Text style={tw`text-gray-400 text-xs mt-1`}>
              {stage === 'analyzing' ? `Processing ${imageUris.length} images...` : 'API-2: Washing Parameter Prediction'}
            </Text>

            {/* Pipeline Steps */}
            <View style={tw`flex-row items-center mt-5 gap-2`}>
              <View style={tw`items-center`}>
                <View style={[
                  tw`w-8 h-8 rounded-full items-center justify-center`,
                  { backgroundColor: stage === 'analyzing' ? '#2A7FFF' : '#00C9A7' }
                ]}>
                  <Text style={tw`text-white font-bold text-xs`}>1</Text>
                </View>
                <Text style={tw`text-xs text-gray-500 mt-1`}>Fabric</Text>
              </View>
              <ArrowRight size={16} color={isDarkMode ? '#334155' : '#CBD5E1'} />
              <View style={tw`items-center`}>
                <View style={[
                  tw`w-8 h-8 rounded-full items-center justify-center`,
                  { backgroundColor: stage === 'predicting' ? '#2A7FFF' : (isDarkMode ? '#334155' : '#E2E8F0') }
                ]}>
                  <Text style={[tw`font-bold text-xs`, { color: stage === 'predicting' ? '#FFF' : '#94A3B8' }]}>2</Text>
                </View>
                <Text style={tw`text-xs text-gray-500 mt-1`}>Wash</Text>
              </View>
            </View>
          </View>
        )}

        {/* Error State */}
        {stage === 'error' && (
          <View style={tw`mt-5 ${isDarkMode ? 'bg-red-900/20 border-red-800/50' : 'bg-red-50 border-red-200'} rounded-2xl p-4`}>
            <Text style={tw`text-red-600 font-bold`}>Analysis Failed</Text>
            <Text style={tw`${isDarkMode ? 'text-red-400' : 'text-red-500'} text-sm mt-1`}>{errorMsg}</Text>
            <TouchableOpacity onPress={reset} style={tw`mt-3`}>
              <Text style={tw`text-primary font-bold`}>Try Again</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Results */}
        {analysis && stage === 'done' && (
          <>
            {/* Dominant Fabric Banner */}
            {analysis.fabrics.length > 0 && (
              <View style={tw`${isDarkMode ? 'bg-primary/20 border-primary/30' : 'bg-primary/8 border-primary/20'} rounded-2xl p-4 my-4 flex-row items-center gap-3`}>
                <Sparkles size={20} color="#2A7FFF" />
                <View style={tw`flex-1`}>
                  <Text style={tw`text-primary font-bold`}>Dominant Fabric</Text>
                  <Text style={tw`${isDarkMode ? 'text-slate-300' : 'text-gray-600'} text-sm`}>
                    {analysis.fabrics[0].name} — {Math.round(analysis.fabrics[0].confidence * 100)}% confidence
                  </Text>
                </View>
              </View>
            )}

            {/* Recommended Cycle Name */}
            <View style={tw`${isDarkMode ? 'bg-secondary/20 border-secondary/30' : 'bg-secondary/10 border-secondary/20'} rounded-2xl p-4 mb-5`}>
              <Text style={tw`text-gray-500 text-xs font-medium uppercase tracking-wider`}>
                Recommended Cycle
              </Text>
              <Text style={tw`${isDarkMode ? 'text-white' : 'text-gray-800'} text-lg font-bold mt-1`}>
                {analysis.recommendedCycle}
              </Text>
            </View>

            {/* Fabric List */}
            <Text style={tw`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'} mb-3`}>Detected Fabrics ({analysis.fabrics.length})</Text>
            {analysis.fabrics.map((fabric, index) => (
              <FabricCard
                key={`${fabric.name}-${index}`}
                fabric={fabric}
                isDominant={index === 0}
              />
            ))}

            {/* Cycle Parameters */}
            {cycle && (
              <View style={tw`mt-5`}>
                <CycleCard cycle={cycle} title="AI Predicted Wash Params" />
                
                <TouchableOpacity
                  onPress={() => navigation.navigate('Dashboard')}
                  style={tw`mt-6 bg-primary py-4 rounded-2xl flex-row items-center justify-center gap-2 shadow-lg`}
                >
                  <ArrowRight size={20} color="white" />
                  <Text style={tw`text-white font-bold text-lg`}>Go to Dashboard to Start</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  onPress={() => navigation.navigate('ManualOverride')}
                  style={tw`mt-4 bg-secondary/10 border-2 border-dashed border-secondary py-4 rounded-xl flex-row items-center justify-center gap-2`}
                >
                  <Plus size={18} color="#00C9A7" />
                  <Text style={tw`text-secondary font-bold text-base`}>Customize AI Settings</Text>
                </TouchableOpacity>
              </View>
            )}
          </>
        )}
      </View>
    </ScrollView>
  );
}
