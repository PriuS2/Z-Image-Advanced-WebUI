import { memo, useState, useCallback } from 'react'
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Languages, Sparkles, Loader2, X } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

// 노드 데이터 타입 정의
interface PromptNodeData {
  prompt?: string
  promptKo?: string
}

function PromptNodeComponent({ id, selected, data }: NodeProps<PromptNodeData>) {
  const { deleteElements, setNodes } = useReactFlow()
  const [isHovered, setIsHovered] = useState(false)
  const { t } = useTranslation()
  const { error: toastError } = useToast()
  
  const [isTranslating, setIsTranslating] = useState(false)
  const [isEnhancing, setIsEnhancing] = useState(false)

  // 노드 자체의 데이터 (글로벌 스토어 대신)
  const prompt = data?.prompt || ''
  const promptKo = data?.promptKo || ''

  // 노드 데이터 업데이트 함수
  const updateNodeData = useCallback((newData: Partial<PromptNodeData>) => {
    setNodes((nodes) =>
      nodes.map((node) =>
        node.id === id
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    )
  }, [id, setNodes])

  const handleTranslate = async () => {
    if (!promptKo) return
    
    setIsTranslating(true)
    try {
      const result = await generationApi.translate(promptKo)
      updateNodeData({ prompt: result.translated })
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsTranslating(false)
    }
  }

  const handleEnhance = async () => {
    if (!prompt) return
    
    setIsEnhancing(true)
    try {
      const result = await generationApi.enhance(prompt)
      updateNodeData({ prompt: result.enhanced })
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsEnhancing(false)
    }
  }

  return (
    <div 
      className={`min-w-[360px] rounded-lg border bg-card p-4 pl-8 pr-16 relative ${selected ? 'border-primary' : 'border-border'}`}
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
      <div className="mb-3 text-sm font-semibold">{t('nodes.prompt')}</div>
      
      {/* Korean prompt */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.promptKo')}
        </label>
        <textarea
          value={promptKo}
          onChange={(e) => updateNodeData({ promptKo: e.target.value })}
          className="nodrag w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            resize-none focus:outline-none focus:ring-2 focus:ring-ring"
          rows={2}
          placeholder="한국어 프롬프트 입력..."
        />
        <button
          onClick={handleTranslate}
          disabled={isTranslating || !promptKo}
          className="nodrag mt-1 flex items-center gap-1 rounded px-2 py-1 text-xs
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
          value={prompt}
          onChange={(e) => updateNodeData({ prompt: e.target.value })}
          className="nodrag w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            resize-none focus:outline-none focus:ring-2 focus:ring-ring"
          rows={3}
          placeholder="Enter prompt in English..."
        />
        <button
          onClick={handleEnhance}
          disabled={isEnhancing || !prompt}
          className="nodrag mt-1 flex items-center gap-1 rounded px-2 py-1 text-xs
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
        className="!bg-green-500 !w-3 !h-3"
      />
      <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-green-500 font-medium">
        Prompt
      </div>
    </div>
  )
}

export const PromptNode = memo(PromptNodeComponent)
