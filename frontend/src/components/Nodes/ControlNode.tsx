import { memo, useState, useRef, useCallback, useEffect } from 'react'
import { Handle, Position, NodeProps, useReactFlow, useStore } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Loader2, Upload, X, Link } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useGenerationStore } from '../../stores/generationStore'
import { useToast } from '../../hooks/useToast'

const controlTypes = ['canny', 'hed', 'depth', 'pose', 'mlsd']

function ControlNodeComponent({ id, selected, data }: NodeProps) {
  const { t } = useTranslation()
  const { setParams } = useGenerationStore()
  const { error: toastError, success: toastSuccess } = useToast()
  const { deleteElements, setNodes, getNode } = useReactFlow()
  
  const [controlType, setControlType] = useState(data?.controlType || 'canny')
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractedImage, setExtractedImage] = useState<string | null>(data?.controlImagePath || null)
  const [sourceImage, setSourceImage] = useState<string | null>(null)
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [isHovered, setIsHovered] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Subscribe to edges changes to detect connections
  const edges = useStore((state) => state.edges)
  
  // Track the connected source node ID
  const connectedSourceId = edges.find(
    (e) => e.target === id && e.targetHandle === 'input'
  )?.source || null
  
  // Subscribe to the specific connected node's imagePreview using nodeInternals
  const connectedImagePreview = useStore((state) => {
    if (!connectedSourceId) return null
    const sourceNode = state.nodeInternals.get(connectedSourceId)
    return (sourceNode?.data?.imagePreview as string) || null
  })
  
  // Update node data so GenerateNode can access it
  const updateNodeData = useCallback((newData: Record<string, unknown>) => {
    setNodes((nodes) =>
      nodes.map((node) =>
        node.id === id
          ? { ...node, data: { ...node.data, ...newData } }
          : node
      )
    )
  }, [id, setNodes])
  
  // Update connection state
  useEffect(() => {
    setIsConnected(!!connectedSourceId)
  }, [connectedSourceId])
  
  // Update source image when connected node's image changes
  useEffect(() => {
    if (connectedSourceId && connectedImagePreview) {
      // Only update if different to avoid unnecessary re-renders
      if (sourceImage !== connectedImagePreview) {
        setSourceImage(connectedImagePreview)
        // Get the source file and dimensions from the node directly
        const sourceNode = getNode(connectedSourceId)
        if (sourceNode?.data?.sourceFile) {
          setSourceFile(sourceNode.data.sourceFile as File)
        }
        // Also copy image dimensions from connected node
        if (sourceNode?.data?.imageWidth && sourceNode?.data?.imageHeight) {
          updateNodeData({
            imageWidth: sourceNode.data.imageWidth as number,
            imageHeight: sourceNode.data.imageHeight as number,
          })
        }
      }
    }
  }, [connectedSourceId, connectedImagePreview, sourceImage, getNode, updateNodeData])

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSourceFile(file)
      const reader = new FileReader()
      reader.onload = (event) => {
        const previewUrl = event.target?.result as string
        setSourceImage(previewUrl)
        
        // Get image dimensions
        const img = new window.Image()
        img.onload = () => {
          updateNodeData({
            imageWidth: img.naturalWidth,
            imageHeight: img.naturalHeight,
          })
        }
        img.src = previewUrl
      }
      reader.readAsDataURL(file)
    }
  }

  const handleControlTypeChange = (newType: string) => {
    setControlType(newType)
    updateNodeData({ controlType: newType })
  }

  const handleExtract = async () => {
    if (!sourceFile) {
      toastError(t('common.error'), 'Please upload an image first')
      return
    }

    setIsExtracting(true)
    try {
      const result = await generationApi.extractControl(sourceFile, controlType)
      setExtractedImage(result.control_image_path)
      
      // Update node data for GenerateNode to access
      updateNodeData({ 
        controlImagePath: result.control_image_path,
        controlType: controlType,
      })
      
      // Also update global params (for backwards compatibility)
      setParams({ controlType, controlImagePath: result.control_image_path })
      toastSuccess(t('common.success'), 'Control image extracted')
    } catch (err) {
      toastError(t('common.error'), String(err))
    } finally {
      setIsExtracting(false)
    }
  }

  const clearImage = () => {
    setSourceImage(null)
    setSourceFile(null)
    setExtractedImage(null)
    updateNodeData({ controlImagePath: null, imageWidth: null, imageHeight: null })
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div 
      className={`min-w-[320px] rounded-lg border bg-card p-4 pl-12 pr-16 relative ${selected ? 'border-primary' : 'border-border'}`}
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
        className={`!w-3 !h-3 ${isConnected ? '!bg-green-500' : '!bg-blue-500'}`}
      />
      <div className={`absolute left-2 top-[50%] -translate-y-1/2 text-[9px] font-medium flex items-center gap-1 ${isConnected ? 'text-green-500' : 'text-blue-500'}`}>
        {isConnected && <Link className="h-2.5 w-2.5" />}
        Image
      </div>
      
      <div className="mb-3 text-sm font-semibold">{t('nodes.control')}</div>
      
      {/* Source image upload */}
      <div className="mb-3">
        {sourceImage ? (
          <div className="relative">
            <img
              src={sourceImage}
              alt="Source"
              className="w-full rounded-md object-cover max-h-24"
            />
            {isConnected ? (
              <div className="mt-1 text-[10px] text-green-500 flex items-center gap-1">
                <Link className="h-2.5 w-2.5" />
                Connected
              </div>
            ) : (
              <button
                onClick={clearImage}
                className="nodrag absolute -right-2 -top-2 rounded-full bg-destructive p-1 text-destructive-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        ) : (
          <div className="relative">
            {isConnected ? (
              <div className="flex flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed
                border-green-500/50 p-3 text-center">
                <Link className="h-5 w-5 text-green-500" />
                <span className="text-xs text-green-500">Waiting for image...</span>
              </div>
            ) : (
              <>
                <div className="flex flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed
                  border-muted-foreground/25 p-3 text-center hover:border-primary/50 transition-colors cursor-pointer">
                  <Upload className="h-5 w-5 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">{t('generate.uploadImage')}</span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="nodrag absolute inset-0 cursor-pointer opacity-0"
                />
              </>
            )}
          </div>
        )}
      </div>
      
      {/* Control type selector */}
      <div className="mb-3">
        <label className="mb-1 block text-xs text-muted-foreground">
          {t('generate.controlType')}
        </label>
        <select
          value={controlType}
          onChange={(e) => handleControlTypeChange(e.target.value)}
          className="nodrag w-full rounded-md border border-input bg-background px-3 py-2 text-sm
            focus:outline-none focus:ring-2 focus:ring-ring"
        >
          {controlTypes.map((type) => (
            <option key={type} value={type}>
              {type.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      
      {/* Extract button */}
      <button
        onClick={handleExtract}
        disabled={isExtracting || !sourceImage}
        className="nodrag w-full flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm
          bg-secondary text-secondary-foreground hover:bg-secondary/80
          disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isExtracting && <Loader2 className="h-4 w-4 animate-spin" />}
        {t('generate.extractControl')}
      </button>
      
      {/* Preview of extracted control */}
      {extractedImage && (
        <div className="mt-3">
          <label className="mb-1 block text-xs text-muted-foreground">Extracted</label>
          <img
            src={extractedImage}
            alt="Control"
            className="w-full max-w-[400px] max-h-[400px] rounded-md object-contain"
          />
        </div>
      )}
      
      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="!bg-orange-500 !w-3 !h-3"
      />
      <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-orange-500 font-medium">
        Control
      </div>
    </div>
  )
}

export const ControlNode = memo(ControlNodeComponent)
