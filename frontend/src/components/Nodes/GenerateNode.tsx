import { memo, useState } from 'react'
import { Handle, Position, NodeProps } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Play, Loader2 } from 'lucide-react'
import { useGenerationStore } from '../../stores/generationStore'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

function GenerateNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { params, progress, startGeneration, setLastGeneratedImage } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()

  const handleGenerate = async () => {
    if (!params.prompt) {
      toastError('Prompt is required')
      return
    }

    try {
      const task = await generationApi.generate({
        prompt: params.prompt,
        prompt_ko: params.promptKo,
        width: params.width,
        height: params.height,
        num_inference_steps: params.steps,
        seed: params.seed,
        control_context_scale: params.controlScale,
        sampler: params.sampler,
        control_type: params.controlType,
      })
      
      startGeneration(task.id)
      toastSuccess(t('common.success'), `Task ${task.id} started`)
    } catch (err) {
      toastError(t('errors.generationFailed'), String(err))
    }
  }

  const isGenerating = progress.isGenerating

  return (
    <div className={`min-w-[180px] rounded-lg border bg-card p-4 ${selected ? 'border-primary' : 'border-border'}`}>
      {/* Input handles */}
      <Handle
        type="target"
        position={Position.Left}
        id="prompt"
        style={{ top: '30%' }}
        className="!bg-primary !w-3 !h-3"
      />
      <Handle
        type="target"
        position={Position.Left}
        id="params"
        style={{ top: '50%' }}
        className="!bg-primary !w-3 !h-3"
      />
      <Handle
        type="target"
        position={Position.Left}
        id="control"
        style={{ top: '70%' }}
        className="!bg-primary !w-3 !h-3"
      />
      
      <div className="mb-3 text-sm font-semibold">{t('nodes.generate')}</div>
      
      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className={`w-full flex items-center justify-center gap-2 rounded-md px-4 py-3 text-sm font-medium
          transition-all duration-200
          ${isGenerating 
            ? 'bg-primary/50 cursor-not-allowed' 
            : 'bg-primary text-primary-foreground hover:bg-primary/90 hover:scale-105'
          }`}
      >
        {isGenerating ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            <span>{progress.progress}%</span>
          </>
        ) : (
          <>
            <Play className="h-5 w-5" />
            {t('generate.generate')}
          </>
        )}
      </button>
      
      {/* Progress indicator */}
      {isGenerating && (
        <div className="mt-3">
          <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
            <div 
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progress.progress}%` }}
            />
          </div>
          <div className="mt-1 text-center text-xs text-muted-foreground">
            Step {progress.currentStep}/{progress.totalSteps}
          </div>
        </div>
      )}
      
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

export const GenerateNode = memo(GenerateNodeComponent)
