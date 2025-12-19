import { memo, useState, useCallback, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { Handle, Position, NodeProps, useReactFlow, useStore } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Paintbrush, Upload, X, Loader2, Link } from 'lucide-react'
import { MaskEditor } from '../MaskEditor/MaskEditor'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

// Helper function to convert base64 data URL to File
function dataURLtoFile(dataUrl: string, filename: string): File {
  const arr = dataUrl.split(',')
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png'
  const bstr = atob(arr[1])
  let n = bstr.length
  const u8arr = new Uint8Array(n)
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n)
  }
  return new File([u8arr], filename, { type: mime })
}

function MaskNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { deleteElements, setNodes } = useReactFlow()
  const { error: toastError, success: toastSuccess } = useToast()
  const [mode, setMode] = useState<'draw' | 'upload'>('draw')
  const [sourceImage, setSourceImage] = useState<string | null>(null)
  const [maskImage, setMaskImage] = useState<string | null>(null)
  const [showEditor, setShowEditor] = useState(false)
  const [isHovered, setIsHovered] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  
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
  
  // Subscribe to imagePath using nodeInternals
  const connectedImagePath = useStore((state) => {
    if (!connectedSourceId) return null
    const sourceNode = state.nodeInternals.get(connectedSourceId)
    return (sourceNode?.data?.imagePath as string) || null
  })
  
  // Track if we've already set the original image path to avoid infinite loops
  const lastSetImagePathRef = useRef<string | null>(null)
  
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
      }
      // Use connected image path as original image (only if changed)
      if (connectedImagePath && connectedImagePath !== lastSetImagePathRef.current) {
        lastSetImagePathRef.current = connectedImagePath
        setNodes((nodes) =>
          nodes.map((node) =>
            node.id === id
              ? { ...node, data: { ...node.data, originalImagePath: connectedImagePath } }
              : node
          )
        )
      }
    }
  }, [connectedSourceId, connectedImagePreview, connectedImagePath, sourceImage, id, setNodes])

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

  // Upload mask image to server and update node data
  const uploadMaskToServer = useCallback(async (maskDataUrl: string) => {
    setIsUploading(true)
    try {
      const file = dataURLtoFile(maskDataUrl, `mask_${Date.now()}.png`)
      const result = await generationApi.uploadImage(file)
      updateNodeData({ maskImagePath: result.path })
      toastSuccess(t('common.success'), 'Mask uploaded')
    } catch (err) {
      toastError(t('common.error'), String(err))
      updateNodeData({ maskImagePath: null })
    } finally {
      setIsUploading(false)
    }
  }, [updateNodeData, toastSuccess, toastError, t])

  // Upload source image as original image for inpainting
  const uploadOriginalImage = useCallback(async (file: File) => {
    try {
      const result = await generationApi.uploadImage(file)
      updateNodeData({ originalImagePath: result.path })
    } catch (err) {
      toastError(t('common.error'), String(err))
      updateNodeData({ originalImagePath: null })
    }
  }, [updateNodeData, toastError, t])

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSourceImage(event.target?.result as string)
      }
      reader.readAsDataURL(file)
      
      // Upload source image as original image for inpainting
      await uploadOriginalImage(file)
    }
  }

  const handleMaskUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = async (event) => {
        const dataUrl = event.target?.result as string
        setMaskImage(dataUrl)
        // Upload mask to server
        await uploadMaskToServer(dataUrl)
      }
      reader.readAsDataURL(file)
    }
  }

  return (
    <>
      <div 
        className={`min-w-[260px] rounded-lg border bg-card p-4 pl-12 pr-14 relative ${selected ? 'border-primary' : 'border-border'}`}
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
        
        <div className="mb-3 text-sm font-semibold">{t('nodes.mask')}</div>
        
        {/* Mode selector */}
        <div className="mb-3 flex gap-1">
          <button
            onClick={() => setMode('draw')}
            className={`nodrag flex-1 flex items-center justify-center gap-1 rounded-md px-2 py-1.5 text-xs transition-colors
              ${mode === 'draw' 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-secondary hover:bg-secondary/80'
              }`}
          >
            <Paintbrush className="h-3 w-3" />
            Draw
          </button>
          <button
            onClick={() => setMode('upload')}
            className={`nodrag flex-1 flex items-center justify-center gap-1 rounded-md px-2 py-1.5 text-xs transition-colors
              ${mode === 'upload' 
                ? 'bg-primary text-primary-foreground' 
                : 'bg-secondary hover:bg-secondary/80'
              }`}
          >
            <Upload className="h-3 w-3" />
            Upload
          </button>
        </div>
        
        {mode === 'draw' ? (
          <>
            {sourceImage ? (
              <div className="relative">
                {/* Image preview with mask overlay */}
                <div 
                  className="relative w-full rounded-md overflow-hidden cursor-pointer bg-black/20"
                  onClick={() => setShowEditor(true)}
                >
                  {/* Original image - object-contain to show full image */}
                  <img
                    src={sourceImage}
                    alt="Source"
                    className="w-full object-contain"
                    style={{ maxHeight: '150px' }}
                  />
                  {/* Mask overlay - red background masked by white mask image */}
                  {maskImage && (
                    <div 
                      className="absolute inset-0 bg-red-500"
                      style={{
                        opacity: 0.5,
                        maskImage: `url(${maskImage})`,
                        WebkitMaskImage: `url(${maskImage})`,
                        maskSize: 'contain',
                        WebkitMaskSize: 'contain',
                        maskRepeat: 'no-repeat',
                        WebkitMaskRepeat: 'no-repeat',
                        maskPosition: 'center',
                        WebkitMaskPosition: 'center',
                      }}
                    />
                  )}
                  {/* Mask indicator */}
                  {maskImage && (
                    <div className="absolute bottom-1 right-1 bg-red-500/80 text-white text-[9px] px-1.5 py-0.5 rounded">
                      Masked
                    </div>
                  )}
                </div>
                {isConnected ? (
                  <div className="mt-1 text-[10px] text-green-500 flex items-center gap-1">
                    <Link className="h-2.5 w-2.5" />
                    Connected
                  </div>
                ) : (
                  <button
                    onClick={() => {
                      setSourceImage(null)
                      setMaskImage(null)
                      updateNodeData({ originalImagePath: null, maskImagePath: null })
                    }}
                    className="nodrag absolute -right-2 -top-2 rounded-full bg-destructive p-1"
                  >
                    <X className="h-3 w-3 text-white" />
                  </button>
                )}
                <button
                  onClick={() => setShowEditor(true)}
                  className="nodrag mt-2 w-full rounded-md bg-primary py-1.5 text-xs text-primary-foreground"
                >
                  {maskImage ? 'Edit Mask' : 'Draw Mask'}
                </button>
              </div>
            ) : (
              <div className="relative">
                {isConnected ? (
                  <div className="flex flex-col items-center justify-center gap-1 rounded-md border-2 border-dashed
                    border-green-500/50 p-4 text-center">
                    <Link className="h-6 w-6 text-green-500" />
                    <span className="text-xs text-green-500">Waiting for image...</span>
                  </div>
                ) : (
                  <>
                    <div className="flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed
                      border-muted-foreground/25 p-4 text-center">
                      <Upload className="h-6 w-6 text-muted-foreground" />
                      <span className="text-xs text-muted-foreground">Upload image to draw mask</span>
                    </div>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="nodrag absolute inset-0 cursor-pointer opacity-0"
                    />
                  </>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="relative">
            {maskImage ? (
              <div className="relative">
                <img
                  src={maskImage}
                  alt="Mask"
                  className="w-full rounded-md object-cover max-h-32"
                />
                <button
                  onClick={() => {
                    setMaskImage(null)
                    updateNodeData({ maskImagePath: null })
                  }}
                  className="nodrag absolute -right-2 -top-2 rounded-full bg-destructive p-1"
                >
                  <X className="h-3 w-3 text-white" />
                </button>
              </div>
            ) : (
              <>
                <div className="flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed
                  border-muted-foreground/25 p-4 text-center">
                  <Upload className="h-6 w-6 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">Upload mask image</span>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleMaskUpload}
                  className="nodrag absolute inset-0 cursor-pointer opacity-0"
                />
              </>
            )}
          </div>
        )}
        
        {/* Output handle */}
        <Handle
          type="source"
          position={Position.Right}
          id="mask"
          className="!bg-pink-500 !w-3 !h-3"
        />
        <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-pink-500 font-medium">
          Mask
        </div>
      </div>

      {/* Mask editor modal - rendered via Portal to escape ReactFlow */}
      {showEditor && sourceImage && createPortal(
        <div 
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 p-4"
          onMouseDown={(e) => e.stopPropagation()}
          onMouseMove={(e) => e.stopPropagation()}
          onMouseUp={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
          onPointerDown={(e) => e.stopPropagation()}
          onPointerMove={(e) => e.stopPropagation()}
          onPointerUp={(e) => e.stopPropagation()}
          onWheel={(e) => e.stopPropagation()}
          onDrag={(e) => e.stopPropagation()}
          onDragStart={(e) => e.stopPropagation()}
        >
          <div 
            className="max-w-4xl w-full max-h-[90vh] overflow-auto rounded-lg border border-border bg-card p-6"
            onMouseDown={(e) => e.stopPropagation()}
            onMouseMove={(e) => e.stopPropagation()}
            onMouseUp={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Mask Editor</h3>
              <button
                onClick={() => setShowEditor(false)}
                className="rounded p-1 hover:bg-accent"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <MaskEditor
              image={sourceImage}
              initialMask={maskImage}
              onMaskChange={(dataUrl) => setMaskImage(dataUrl)}
            />
            
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => setShowEditor(false)}
                className="rounded-md bg-secondary px-4 py-2 text-sm hover:bg-secondary/80"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  if (maskImage) {
                    await uploadMaskToServer(maskImage)
                  }
                  setShowEditor(false)
                }}
                disabled={isUploading}
                className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90 
                  disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isUploading && <Loader2 className="h-4 w-4 animate-spin" />}
                Apply Mask
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  )
}

export const MaskNode = memo(MaskNodeComponent)
