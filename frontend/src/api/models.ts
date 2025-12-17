import { apiClient, handleApiError } from './client'

export interface ModelInfo {
  name: string
  path: string
  size?: number
  loaded: boolean
  type: string
}

export interface ModelStatus {
  base_model_loaded: boolean
  controlnet_loaded: boolean
  current_model?: string
  vram_usage?: number
  gpu_memory_mode: string
}

export interface ModelLoadRequest {
  model_path: string
  gpu_memory_mode?: string
  weight_dtype?: string
}

export interface ModelDownloadRequest {
  repo_id: string
  filename?: string
  destination: string
}

export interface DownloadProgress {
  status: 'idle' | 'downloading' | 'completed' | 'error'
  progress: number
  message: string
  repo_id: string
}

export const modelsApi = {
  list: async (modelType?: string): Promise<ModelInfo[]> => {
    try {
      const params = modelType ? { model_type: modelType } : {}
      const response = await apiClient.get<ModelInfo[]>('/models/list', { params })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  getStatus: async (): Promise<ModelStatus> => {
    try {
      const response = await apiClient.get<ModelStatus>('/models/status')
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  load: async (request: ModelLoadRequest): Promise<{ message: string }> => {
    try {
      const response = await apiClient.post<{ message: string }>('/models/load', request)
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  unload: async (): Promise<{ message: string }> => {
    try {
      const response = await apiClient.post<{ message: string }>('/models/unload')
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  download: async (request: ModelDownloadRequest): Promise<{ message: string; status: string }> => {
    try {
      const response = await apiClient.post<{ message: string; status: string }>('/models/download', request)
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  getDownloadProgress: async (): Promise<DownloadProgress> => {
    try {
      const response = await apiClient.get<DownloadProgress>('/models/download/progress')
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  listLoras: async (): Promise<ModelInfo[]> => {
    try {
      const response = await apiClient.get<ModelInfo[]>('/models/loras')
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  applyLora: async (loraPath: string, weight: number = 0.8): Promise<{ message: string }> => {
    try {
      const response = await apiClient.post<{ message: string }>('/models/loras/apply', null, {
        params: { lora_path: loraPath, weight }
      })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },
}
