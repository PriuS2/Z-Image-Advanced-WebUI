import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Languages, Sparkles, Loader2 } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function PromptNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { params, setParams } = useGenerationStore()
  const { error: toastError } = useToast()
  
  const [isTranslating, setIsTranslating] = useState(false)
  const [isEnhancing, setIsEnhancing] = useState(false)

  const handleTranslate = async () => {
    if (!params.promptKo) return
    
    setIsTranslating(true)
    try {
      const result = await generationApi.translate(params.promptKo)
      setParams({ prompt: result.translated })
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsTranslating(false)
    }
  }

  const handleEnhance = async () => {
    if (!params.prompt) return
    
    setIsEnhancing(true)
    try {
      const result = await generationApi.enhance(params.prompt)
      setParams({ prompt: result.enhanced })
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsEnhancing(false)
    }
  }

  return (
    <div className={`min-w-[300px] rounded-lg border bg-card p-4 ${selected ? 'border-primary' : 'border-border'}`}>
      <div className="mb-3 text-sm font-semibold">{t('nodes.prompt')}</div>
      
      {/* Korean prompt */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.promptKo')}
        </label>
        <textarea
          value={params.promptKo}
          onChange={(e) => setParams({ promptKo: e.target.value })}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            resize-none focus:outline-none focus:ring-2 focus:ring-ring"
          rows={2}
          placeholder="한국어 프롬프트 입력..."
        />
        <button
          onClick={handleTranslate}
          disabled={isTranslating || !params.promptKo}
          className="mt-1 flex items-center gap-1 rounded px-2 py-1 text-xs
            bg-secondary text-secondary-foreground hover:bg-secondary/80
            disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isTranslating ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Languages className="h-3 w-3" />
          )}
          {t('generate.translate')}
        </button>
      </div>
      
      {/* English prompt */}
      <div>
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.promptEn')}
        </label>
        <textarea
          value={params.prompt}
          onChange={(e) => setParams({ prompt: e.target.value })}
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            resize-none focus:outline-none focus:ring-2 focus:ring-ring"
          rows={3}
          placeholder="Enter prompt in English..."
        />
        <button
          onClick={handleEnhance}
          disabled={isEnhancing || !params.prompt}
          className="mt-1 flex items-center gap-1 rounded px-2 py-1 text-xs
            bg-primary text-primary-foreground hover:bg-primary/90
            disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isEnhancing ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Sparkles className="h-3 w-3" />
          )}
          {t('generate.enhance')}
        </button>
      </div>
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-primary !w-3 !h-3"
      />
    </div>
  )
}

export const PromptNode = memo(PromptNodeComponent)
