import { create } from 'zustand'
import { Node, Edge } from 'reactflow'

interface GenerationParams {
  prompt: string
  promptKo: string
  width: number
  height: number
  steps: number
  seed: number | null
  controlScale: number
  sampler: string
  controlType: string | null
  controlImage: string | null
  controlImagePath: string | null
  maskImage: string | null
}

interface GenerationProgress {
  taskId: number | null
  progress: number
  currentStep: number
  totalSteps: number
  elapsedTime: number
  estimatedRemaining: number
  currentNode: string
  isGenerating: boolean
}

// Node connection state for tracking which nodes are connected to Generate node
interface NodeConnections {
  isControlConnected: boolean
  isMaskConnected: boolean
  controlNodeId: string | null
  maskNodeId: string | null
}

interface GenerationState {
  // Parameters
  params: GenerationParams
  
  // Flow editor
  nodes: Node[]
  edges: Edge[]
  
  // Node connection state
  nodeConnections: NodeConnections
  
  // Progress
  progress: GenerationProgress
  
  // Generated result
  lastGeneratedImage: string | null
  lastGeneratedImageId: number | null
  
  // Actions
  setParams: (params: Partial<GenerationParams>) => void
  setNodes: (nodes: Node[]) => void
  setEdges: (edges: Edge[]) => void
  setNodeConnections: (connections: Partial<NodeConnections>) => void
  updateConnectionsFromEdges: (edges: Edge[], generateNodeId: string) => void
  setProgress: (progress: Partial<GenerationProgress>) => void
  setLastGeneratedImage: (image: string | null, imageId?: number | null) => void
  resetProgress: () => void
  startGeneration: (taskId: number) => void
}

const defaultParams: GenerationParams = {
  prompt: '',
  promptKo: '',
  width: 1024,
  height: 1024,
  steps: 25,
  seed: null,
  controlScale: 0.75,
  sampler: 'Flow',
  controlType: null,
  controlImage: null,
  controlImagePath: null,
  maskImage: null,
}

const defaultProgress: GenerationProgress = {
  taskId: null,
  progress: 0,
  currentStep: 0,
  totalSteps: 0,
  elapsedTime: 0,
  estimatedRemaining: 0,
  currentNode: '',
  isGenerating: false,
}

const defaultNodeConnections: NodeConnections = {
  isControlConnected: false,
  isMaskConnected: false,
  controlNodeId: null,
  maskNodeId: null,
}

export const useGenerationStore = create<GenerationState>()((set) => ({
  params: defaultParams,
  nodes: [],
  edges: [],
  nodeConnections: defaultNodeConnections,
  progress: defaultProgress,
  lastGeneratedImage: null,
  lastGeneratedImageId: null,

  setParams: (newParams) =>
    set((state) => ({
      params: { ...state.params, ...newParams },
    })),

  setNodes: (nodes) => set({ nodes }),

  setEdges: (edges) => set({ edges }),

  setNodeConnections: (connections) =>
    set((state) => ({
      nodeConnections: { ...state.nodeConnections, ...connections },
    })),

  updateConnectionsFromEdges: (edges, generateNodeId) => {
    // Find edges connected to the generate node's control and mask handles
    const controlEdge = edges.find(
      (e) => e.target === generateNodeId && e.targetHandle === 'control'
    )
    const maskEdge = edges.find(
      (e) => e.target === generateNodeId && e.targetHandle === 'mask'
    )

    set({
      nodeConnections: {
        isControlConnected: !!controlEdge,
        isMaskConnected: !!maskEdge,
        controlNodeId: controlEdge?.source || null,
        maskNodeId: maskEdge?.source || null,
      },
    })
  },

  setProgress: (newProgress) =>
    set((state) => ({
      progress: { ...state.progress, ...newProgress },
    })),

  setLastGeneratedImage: (image, imageId = null) => set({ 
    lastGeneratedImage: image,
    lastGeneratedImageId: imageId,
  }),

  resetProgress: () => set({ progress: defaultProgress }),

  startGeneration: (taskId) =>
    set({
      progress: {
        ...defaultProgress,
        taskId,
        isGenerating: true,
      },
    }),
}))
