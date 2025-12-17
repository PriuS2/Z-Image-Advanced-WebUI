import { memo, useState } from 'react'
import { Handle, Position, NodeProps, useReactFlow } from 'reactflow'
import { useTranslation } from 'react-i18next'
import { Paintbrush, Upload, X } from 'lucide-react'
import { MaskEditor } from '../MaskEditor/MaskEditor'

function MaskNodeComponent({ id, selected }: NodeProps) {
  const { t } = useTranslation()
  const { deleteElements } = useReactFlow()
  const [mode, setMode] = useState<'draw' | 'upload'>('draw')
  const [sourceImage, setSourceImage] = useState<string | null>(null)
  const [maskImage, setMaskImage] = useState<string | null>(null)
  const [showEditor, setShowEditor] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setSourceImage(event.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleMaskUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setMaskImage(event.target?.result as string)
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
          className="!bg-blue-500 !w-3 !h-3"
        />
        <div className="absolute left-2 top-[50%] -translate-y-1/2 text-[9px] text-blue-500 font-medium">
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
                <img
                  src={sourceImage}
                  alt="Source"
                  className="w-full rounded-md object-cover max-h-32 cursor-pointer"
                  onClick={() => setShowEditor(true)}
                />
                <button
                  onClick={() => setSourceImage(null)}
                  className="nodrag absolute -right-2 -top-2 rounded-full bg-destructive p-1"
                >
                  <X className="h-3 w-3 text-white" />
                </button>
                <button
                  onClick={() => setShowEditor(true)}
                  className="nodrag mt-2 w-full rounded-md bg-primary py-1.5 text-xs text-primary-foreground"
                >
                  Edit Mask
                </button>
              </div>
            ) : (
              <div className="relative">
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
                  onClick={() => setMaskImage(null)}
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
          id="output"
          className="!bg-pink-500 !w-3 !h-3"
        />
        <div className="absolute right-2 top-[50%] -translate-y-1/2 text-[9px] text-pink-500 font-medium">
          Mask
        </div>
      </div>

      {/* Mask editor modal */}
      {showEditor && sourceImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
          <div className="max-w-4xl w-full max-h-[90vh] overflow-auto rounded-lg border border-border bg-card p-6">
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
                onClick={() => setShowEditor(false)}
                className="rounded-md bg-primary px-4 py-2 text-sm text-primary-foreground hover:bg-primary/90"
              >
                Apply Mask
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export const MaskNode = memo(MaskNodeComponent)
