import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Moon, Sun, Globe, Download, Loader2, RefreshCw, Save } from 'lucide-react'
import { useSettingsStore } from '../../stores/settingsStore'
import { useToast } from '../../hooks/useToast'

export function SettingsTab() {
  const { t, i18n } = useTranslation()
  const { settings, setSettings, setTheme, setLanguage } = useSettingsStore()
  const { success: toastSuccess } = useToast()
  
  const [activeSection, setActiveSection] = useState('general')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [isDownloading, setIsDownloading] = useState(false)

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
            
            {/* GPU Memory Mode */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.gpuMemoryMode')}</label>
              <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="model_full_load">Full Load (High VRAM)</option>
                <option value="model_cpu_offload">CPU Offload</option>
                <option value="model_cpu_offload_and_qfloat8">CPU Offload + FP8</option>
                <option value="sequential_cpu_offload">Sequential Offload (Low VRAM)</option>
              </select>
            </div>
            
            {/* Weight Dtype */}
            <div>
              <label className="mb-2 block text-sm font-medium">{t('settings.weightDtype')}</label>
              <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="bfloat16">BFloat16</option>
                <option value="float16">Float16</option>
                <option value="float32">Float32</option>
              </select>
            </div>
            
            {/* Model actions */}
            <div className="flex gap-2">
              <button className="flex items-center gap-2 rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80">
                <RefreshCw className="h-4 w-4" />
                {t('settings.loadModel')}
              </button>
              <button className="flex items-center gap-2 rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80">
                {t('settings.unloadModel')}
              </button>
              <button 
                className="flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm 
                  text-primary-foreground hover:bg-primary/90"
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
