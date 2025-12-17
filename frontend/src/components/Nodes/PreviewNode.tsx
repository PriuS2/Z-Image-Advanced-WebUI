import { memo, useState } from 'react'
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Download, Heart, RotateCcw, Image as ImageIcon, Loader2, X } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { galleryApi } from '../../api/gallery'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function PreviewNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { lastGeneratedImage, lastGeneratedImageId, progress, params, startGeneration } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  const { deleteElements } = useReactFlow()
  
  const [isFavorite, setIsFavorite] = useState(false)
  const [isRegenerating, setIsRegenerating] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

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
    <div 
      className={`min-w-[320px] rounded-lg border bg-card p-4 pl-12 relative ${selected ? 'border-primary' : 'border-border'}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Delete button */}
      {isHovered && (
        <button
          onClick={() => deleteElements({ nodes: [{ id }] })}
          className="absolute -right-2 -top-2 z-10 rounded-full bg-destructive p-1 text-destructive-foreground 
            hover:bg-destructive/80 transition-colors shadow-md"
        >
          <X className="h-3 w-3" />
        </button>
      )}
      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="!bg-blue-500 !w-3 !h-3"
      />
      <div className="absolute left-2 top-[50%] -translate-y-1/2 text-[9px] text-blue-500 font-medium">
        Image
      </div>
      
      <div className="mb-3 flex items-center justify-between">
        <span className="text-sm font-semibold">{t('nodes.preview')}</span>
        
        {lastGeneratedImage && (
          <div className="flex gap-1">
            <button
              onClick={handleDownload}
              className="nodrag rounded p-1 hover:bg-accent transition-colors"
              title={t('gallery.download')}
            >
              <Download className="h-4 w-4" />
            </button>
            <button
              onClick={handleFavorite}
              className="nodrag rounded p-1 hover:bg-accent transition-colors"
              title={t('gallery.favorite')}
            >
              <Heart className={`h-4 w-4 ${isFavorite ? 'fill-red-500 text-red-500' : ''}`} />
            </button>
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="nodrag rounded p-1 hover:bg-accent transition-colors disabled:opacity-50"
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
