import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Moon, Sun, Globe, Download, Loader2, RefreshCw, Save, Trash2 } from 'lucide-react'
import { useSettingsStore } from '../../stores/settingsStore'
import { useToast } from '../../hooks/useToast'
import { modelsApi, ModelStatus, DownloadProgress } from '../../api/models'

export function SettingsTab() {
  const { t, i18n } = useTranslation()
  const { settings, setSettings, setTheme, setLanguage } = useSettingsStore()
  const { success: toastSuccess, error: toastError } = useToast()
  
  const [activeSection, setActiveSection] = useState('general')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [isDownloading, setIsDownloading] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isUnloading, setIsUnloading] = useState(false)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [gpuMemoryMode, setGpuMemoryMode] = useState('model_cpu_offload_and_qfloat8')
  const [weightDtype, setWeightDtype] = useState('bfloat16')
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)

  const sections = [
    { id: 'general', label: t('settings.general') },
    { id: 'models', label: t('settings.models') },
    { id: 'llm', label: t('settings.llm') },
  ]

  const handleThemeChange = (theme: 'dark' | 'light') => {
    setTheme(theme)
  }

  const handleLanguageChange = (language: 'ko' | 'en') => {
    setLanguage(language)
    i18n.changeLanguage(language)
  }

  const handleSaveSettings = () => {
    toastSuccess(t('common.success'), 'Settings saved')
  }

  // 모델 상태 가져오기
  const fetchModelStatus = async () => {
    try {
      const status = await modelsApi.getStatus()
      setModelStatus(status)
      if (status.gpu_memory_mode) {
        setGpuMemoryMode(status.gpu_memory_mode)
      }
    } catch (error) {
      console.error('Failed to fetch model status:', error)
    }
  }

  useEffect(() => {
    if (activeSection === 'models') {
      fetchModelStatus()
    }
  }, [activeSection])

  // 모델 로드
  const handleLoadModel = async () => {
    setIsLoading(true)
    try {
      const result = await modelsApi.load({
        model_path: 'default',
        gpu_memory_mode: gpuMemoryMode,
        weight_dtype: weightDtype,
      })
      toastSuccess(t('common.success'), result.message || t('settings.modelLoaded'))
      await fetchModelStatus()
    } catch (error) {
      toastError(t('common.error'), error instanceof Error ? error.message : 'Failed to load model')
    } finally {
      setIsLoading(false)
    }
  }

  // 모델 언로드
  const handleUnloadModel = async () => {
    setIsUnloading(true)
    try {
      const result = await modelsApi.unload()
      toastSuccess(t('common.success'), result.message || t('settings.modelUnloaded'))
      await fetchModelStatus()
    } catch (error) {
      toastError(t('common.error'), error instanceof Error ? error.message : 'Failed to unload model')
    } finally {
      setIsUnloading(false)
    }
  }

  // 다운로드 진행 상태 폴링
  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null

    if (isDownloading) {
      intervalId = setInterval(async () => {
        try {
          const progress = await modelsApi.getDownloadProgress()
          setDownloadProgress(progress)

          if (progress.status === 'completed') {
            setIsDownloading(false)
            toastSuccess(t('common.success'), t('settings.modelDownloaded'))
            if (intervalId) clearInterval(intervalId)
          } else if (progress.status === 'error') {
            setIsDownloading(false)
            toastError(t('common.error'), progress.message)
            if (intervalId) clearInterval(intervalId)
          }
        } catch (error) {
          console.error('Failed to get download progress:', error)
        }
      }, 2000) // Poll every 2 seconds
    }

    return () => {
      if (intervalId) clearInterval(intervalId)
    }
  }, [isDownloading])

  // 모델 다운로드
  const handleDownloadModel = async () => {
    setIsDownloading(true)
    setDownloadProgress({
      status: 'downloading',
      progress: 0,
      message: 'Starting download...',
      repo_id: 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0',
    })
    
    try {
      const result = await modelsApi.download({
        repo_id: 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0',
        destination: 'models/',
      })
      toastSuccess(t('common.success'), result.message)
      // Keep polling for actual completion
    } catch (error) {
      setIsDownloading(false)
      setDownloadProgress(null)
      toastError(t('common.error'), error instanceof Error ? error.message : 'Failed to download model')
    }
  }

  return (
    <div className="flex h-full gap-6">
      {/* Sidebar */}
      <div className="w-48 flex-shrink-0">
        <h2 className="mb-4 text-xl font-bold">{t('settings.title')}</h2>
        
        <nav className="space-y-1">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`w-full rounded-md px-3 py-2 text-left text-sm transition-colors
                ${activeSection === section.id
                  ? 'bg-primary text-primary-foreground'
                  : 'hover:bg-accent'
                }`}
            >
              {section.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      <div className="flex-1 max-w-2xl">
        {activeSection === 'general' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">{t('settings.general')}</h3>
            
            {/* Theme */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.theme')}</label>
              <div className="flex gap-2">
                <button
                  onClick={() => handleThemeChange('light')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm transition-colors
                    ${settings.theme === 'light'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary hover:bg-secondary/80'
                    }`}
                >
                  <Sun className="h-4 w-4" />
                  {t('settings.light')}
                </button>
                <button
                  onClick={() => handleThemeChange('dark')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm transition-colors
                    ${settings.theme === 'dark'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary hover:bg-secondary/80'
                    }`}
                >
                  <Moon className="h-4 w-4" />
                  {t('settings.dark')}
                </button>
              </div>
            </div>
            
            {/* Language */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.language')}</label>
              <div className="flex gap-2">
                <button
                  onClick={() => handleLanguageChange('ko')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm transition-colors
                    ${settings.language === 'ko'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary hover:bg-secondary/80'
                    }`}
                >
                  <Globe className="h-4 w-4" />
                  한국어
                </button>
                <button
                  onClick={() => handleLanguageChange('en')}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm transition-colors
                    ${settings.language === 'en'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary hover:bg-secondary/80'
                    }`}
                >
                  <Globe className="h-4 w-4" />
                  English
                </button>
              </div>
            </div>
            
            {/* Realtime preview */}
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">{t('settings.realtimePreview')}</label>
              <button
                onClick={() => setSettings({ realtimePreview: !settings.realtimePreview })}
                className={`relative h-6 w-11 rounded-full transition-colors
                  ${settings.realtimePreview ? 'bg-primary' : 'bg-muted'}`}
              >
                <div
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform
                    ${settings.realtimePreview ? 'translate-x-5' : 'translate-x-0.5'}`}
                />
              </button>
            </div>
          </div>
        )}

        {activeSection === 'models' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">{t('settings.models')}</h3>
            
            {/* Model Status */}
            {modelStatus && (
              <div className="rounded-md border border-border bg-muted/50 p-3 text-sm">
                <div className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${modelStatus.base_model_loaded ? 'bg-green-500' : 'bg-gray-400'}`} />
                  <span>{t('settings.baseModel')}: {modelStatus.base_model_loaded ? t('settings.loaded') : t('settings.notLoaded')}</span>
                </div>
                {modelStatus.vram_usage !== undefined && modelStatus.vram_usage !== null && (
                  <div className="mt-1 text-muted-foreground">
                    VRAM: {modelStatus.vram_usage.toFixed(2)} GB
                  </div>
                )}
              </div>
            )}

            {/* GPU Memory Mode */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.gpuMemoryMode')}</label>
              <select 
                value={gpuMemoryMode}
                onChange={(e) => setGpuMemoryMode(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="model_full_load">Full Load (High VRAM)</option>
                <option value="model_cpu_offload">CPU Offload</option>
                <option value="model_cpu_offload_and_qfloat8">CPU Offload + FP8</option>
                <option value="sequential_cpu_offload">Sequential Offload (Low VRAM)</option>
              </select>
            </div>
            
            {/* Weight Dtype */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.weightDtype')}</label>
              <select 
                value={weightDtype}
                onChange={(e) => setWeightDtype(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="bfloat16">BFloat16</option>
                <option value="float16">Float16</option>
                <option value="float32">Float32</option>
              </select>
            </div>
            
            {/* Model actions */}
            <div className="flex gap-2">
              <button 
                onClick={handleLoadModel}
                disabled={isLoading}
                className="flex items-center gap-2 rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80 disabled:opacity-50"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                {t('settings.loadModel')}
              </button>
              <button 
                onClick={handleUnloadModel}
                disabled={isUnloading}
                className="flex items-center gap-2 rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80 disabled:opacity-50"
              >
                {isUnloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="h-4 w-4" />
                )}
                {t('settings.unloadModel')}
              </button>
              <button 
                onClick={handleDownloadModel}
                className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm 
                  text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                disabled={isDownloading}
              >
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                {t('settings.downloadModel')}
              </button>
            </div>

            {/* Download progress */}
            {isDownloading && downloadProgress && (
              <div className="rounded-md border border-border bg-muted/50 p-4">
                <div className="flex items-center gap-3 mb-2">
                  <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  <span className="text-sm font-medium">
                    {downloadProgress.status === 'downloading' ? '모델 다운로드 중...' : downloadProgress.message}
                  </span>
                </div>
                {/* Animated indeterminate progress bar */}
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div 
                    className="h-full bg-primary animate-pulse"
                    style={{ 
                      width: downloadProgress.progress > 0 ? `${downloadProgress.progress}%` : '100%',
                      animation: downloadProgress.progress === 0 ? 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite' : 'none'
                    }}
                  />
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <p className="text-xs text-muted-foreground truncate flex-1">
                    {downloadProgress.repo_id}
                  </p>
                  <p className="text-xs text-muted-foreground ml-2">
                    터미널에서 진행 상황 확인
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {activeSection === 'llm' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">{t('settings.llm')}</h3>
            
            {/* Provider */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.llmProvider')}</label>
              <select 
                value={settings.llmPreferences.provider}
                onChange={(e) => setSettings({
                  llmPreferences: { ...settings.llmPreferences, provider: e.target.value }
                })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              >
                <option value="openai">OpenAI</option>
                <option value="claude">Claude (Anthropic)</option>
                <option value="gemini">Gemini (Google)</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </div>
            
            {/* API Key */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.apiKey')}</label>
              <input
                type="password"
                value={llmApiKey}
                onChange={(e) => setLlmApiKey(e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="sk-..."
              />
            </div>
            
            {/* Model */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.model')}</label>
              <input
                type="text"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="gpt-4o-mini"
              />
            </div>
            
            {/* System prompts */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.translatePrompt')}</label>
              <textarea
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-none"
                rows={3}
                placeholder="System prompt for translation..."
              />
              <button className="mt-1 text-xs text-primary hover:underline">
                {t('settings.reset')}
              </button>
            </div>
            
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.enhancePrompt')}</label>
              <textarea
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm resize-none"
                rows={3}
                placeholder="System prompt for enhancement..."
              />
              <button className="mt-1 text-xs text-primary hover:underline">
                {t('settings.reset')}
              </button>
            </div>
          </div>
        )}

        {/* Save button */}
        <div className="mt-8">
          <button
            onClick={handleSaveSettings}
            className="flex items-center gap-2 rounded-md bg-primary px-6 py-2 text-sm
              text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            <Save className="h-4 w-4" />
            {t('settings.save')}
          </button>
        </div>
      </div>
    </div>
  )
}
