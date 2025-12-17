import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Download, Heart, RotateCcw, Image as ImageIcon, Loader2 } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { galleryApi } from '../../api/gallery'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function PreviewNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { lastGeneratedImage, lastGeneratedImageId, progress, params, startGeneration } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  
  const [isFavorite, setIsFavorite] = useState(false)
  const [isRegenerating, setIsRegenerating] = useState(false)

  const handleDownload = () => {
    if (lastGeneratedImage) {
      const link = document.createElement('a')
      link.href = lastGeneratedImage
      link.download = `generated_${Date.now()}.png`
      link.click()
    }
  }

  const handleFavorite = async () => {
    if (lastGeneratedImageId) {
      try {
        const result = await galleryApi.toggleFavorite(lastGeneratedImageId)
        setIsFavorite(result.is_favorite)
        toastSuccess(t('common.success'), result.is_favorite ? 'Added to favorites' : 'Removed from favorites')
      } catch (err) {
        toastError(t('common.error'), String(err))
      }
    }
  }

  const handleRegenerate = async () => {
    if (!params.prompt) return
    
    setIsRegenerating(true)
    try {
      const task = await generationApi.generate({
        prompt: params.prompt,
        prompt_ko: params.promptKo,
        width: params.width,
        height: params.height,
        num_inference_steps: params.steps,
        seed: null, // New random seed
        control_context_scale: params.controlScale,
        sampler: params.sampler,
        control_type: params.controlType,
      })
      startGeneration(task.id)
      toastSuccess(t('common.success'), `Task ${task.id} started`)
    } catch (err) {
      toastError(t('errors.generationFailed'), String(err))
    } finally {
      setIsRegenerating(false)
    }
  }

  return (
    <div className={`min-w-[300px] rounded-lg border bg-card p-4 ${selected ? 'border-primary' : 'border-border'}`}>
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="!bg-primary !w-3 !h-3"
      />
      
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold">{t('nodes.preview')}</span>
        
        {lastGeneratedImage && (
          <div className="flex gap-1">
            <button
              onClick={handleDownload}
              className="rounded p-1 hover:bg-accent transition-colors"
              title={t('gallery.download')}
            >
              <Download className="h-4 w-4" />
            </button>
            <button
              onClick={handleFavorite}
              className="rounded p-1 hover:bg-accent transition-colors"
              title={t('gallery.favorite')}
            >
              <Heart className={`h-4 w-4 ${isFavorite ? 'fill-red-500 text-red-500' : ''}`} />
            </button>
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="rounded p-1 hover:bg-accent transition-colors disabled:opacity-50"
              title={t('gallery.regenerate')}
            >
              {isRegenerating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
              <RotateCcw className="h-4 w-4" />
              )}
            </button>
          </div>
        )}
      </div>
      
      {/* Preview area */}
      <div className="aspect-square w-full rounded-lg border border-border bg-muted/50 overflow-hidden">
        {lastGeneratedImage ? (
          <img
            src={lastGeneratedImage}
            alt="Generated"
            className="h-full w-full object-contain"
          />
        ) : progress.isGenerating ? (
          <div className="h-full w-full flex flex-col items-center justify-center gap-2">
            <div className="relative">
              <div className="h-16 w-16 rounded-full border-4 border-primary/20" />
              <div 
                className="absolute inset-0 h-16 w-16 rounded-full border-4 border-primary border-t-transparent animate-spin"
              />
            </div>
            <span className="text-sm text-muted-foreground">
              {progress.progress}%
            </span>
          </div>
        ) : (
          <div className="h-full w-full flex flex-col items-center justify-center gap-2 text-muted-foreground">
            <ImageIcon className="h-12 w-12" />
            <span className="text-sm">No image yet</span>
          </div>
        )}
      </div>
    </div>
  )
}

export const PreviewNode = memo(PreviewNodeComponent)
