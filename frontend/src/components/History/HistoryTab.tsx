import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Search, Heart, Trash2, Copy, Loader2, History } from 'lucide-react'
import { apiClient, handleApiError } from '../../api/client'
import { useToast } from '../../hooks/useToast'
import { useGenerationStore } from '../../stores/generationStore'

interface PromptHistory {
  id: number
  prompt_ko?: string
  prompt_en?: string
  prompt_enhanced?: string
  created_at: string
  is_favorite: boolean
}

export function HistoryTab() {
  const { t } = useTranslation()
  const { error: toastError, success: toastSuccess } = useToast()
  const { setParams } = useGenerationStore()
  
  const [history, setHistory] = useState<PromptHistory[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [favoriteOnly, setFavoriteOnly] = useState(false)

  const loadHistory = async () => {
    setIsLoading(true)
    try {
      const params: Record<string, unknown> = {
        page: 1,
        page_size: 50,
        favorite_only: favoriteOnly,
      }
      if (search) {
        params.search = search
      }
      
      const response = await apiClient.get<PromptHistory[]>('/history/', { params })
      setHistory(response.data)
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    const debounce = setTimeout(loadHistory, 300)
    return () => clearTimeout(debounce)
  }, [search, favoriteOnly])

  const handleToggleFavorite = async (id: number) => {
    try {
      const response = await apiClient.post<{ is_favorite: boolean }>(`/history/${id}/favorite`)
      setHistory((prev) =>
        prev.map((item) =>
          item.id === id ? { ...item, is_favorite: response.data.is_favorite } : item
        )
      )
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await apiClient.delete(`/history/${id}`)
      setHistory((prev) => prev.filter((item) => item.id !== id))
    } catch (err) {
      toastError(t('errors.networkError'), handleApiError(err))
    }
  }

  const handleUsePrompt = (item: PromptHistory) => {
    setParams({
      promptKo: item.prompt_ko || '',
      prompt: item.prompt_enhanced || item.prompt_en || '',
    })
    toastSuccess(t('common.success'), 'Prompt loaded')
  }

  return (
    <div className="h-full">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">{t('history.title')}</h2>
        
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('history.search')}
              className="rounded-md border border-input bg-background pl-10 pr-4 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          
          <button
            onClick={() => setFavoriteOnly(!favoriteOnly)}
            className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors
              ${favoriteOnly 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
              }`}
          >
            <Heart className={`h-4 w-4 ${favoriteOnly ? 'fill-current' : ''}`} />
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : history.length === 0 ? (
        <div className="flex h-64 flex-col items-center justify-center gap-4 text-muted-foreground">
          <History className="h-16 w-16" />
          <p>{t('history.noHistory')}</p>
        </div>
      ) : (
        <div className="space-y-2">
          {history.map((item) => (
            <div
              key={item.id}
              className="group rounded-lg border border-border bg-card p-4 hover:border-primary/50 transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  {item.prompt_ko && (
                    <p className="text-sm text-muted-foreground truncate mb-1">
                      🇰🇷 {item.prompt_ko}
                    </p>
                  )}
                  <p className="text-sm truncate">
                    🇺🇸 {item.prompt_enhanced || item.prompt_en || '-'}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
                
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => handleUsePrompt(item)}
                    className="rounded p-1.5 hover:bg-accent transition-colors"
                    title="Use this prompt"
                  >
                    <Copy className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => handleToggleFavorite(item.id)}
                    className="rounded p-1.5 hover:bg-accent transition-colors"
                  >
                    <Heart className={`h-4 w-4 ${item.is_favorite ? 'fill-red-500 text-red-500' : ''}`} />
                  </button>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="rounded p-1.5 hover:bg-destructive/10 hover:text-destructive transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
