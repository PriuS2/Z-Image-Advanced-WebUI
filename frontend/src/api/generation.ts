import { apiClient, handleApiError } from './client'

interface GenerationRequest {
  prompt: string
  prompt_ko?: string
  width?: number
  height?: number
  num_inference_steps?: number
  guidance_scale?: number
  control_context_scale?: number
  seed?: number | null
  sampler?: string
  control_type?: string | null
  control_image_path?: string | null
  mask_image_path?: string | null
}

interface TaskResponse {
  id: number
  status: string
  progress: number
  created_at: string
  result_path?: string
  error_message?: string
}

interface TranslateResponse {
  original: string
  translated: string
  provider: string
}

interface EnhanceResponse {
  original: string
  enhanced: string
  provider: string
}

interface ControlExtractionResponse {
  control_image_path: string
  control_type: string
}

interface UploadResponse {
  path: string
  filename: string
}

export const generationApi = {
  generate: async (request: GenerationRequest): Promise<TaskResponse> => {
    try {
      const response = await apiClient.post<TaskResponse>('/generation/generate', request)
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  extractControl: async (
    imageFile: File,
    controlType: string,
    lowThreshold: number = 100,
    highThreshold: number = 200
  ): Promise<ControlExtractionResponse> => {
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      formData.append('request', JSON.stringify({
        control_type: controlType,
        low_threshold: lowThreshold,
        high_threshold: highThreshold,
      }))
      
      const response = await apiClient.post<ControlExtractionResponse>(
        '/generation/extract-control',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  uploadImage: async (imageFile: File): Promise<UploadResponse> => {
    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      
      const response = await apiClient.post<UploadResponse>(
        '/generation/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  getTaskStatus: async (taskId: number): Promise<TaskResponse> => {
    try {
      const response = await apiClient.get<TaskResponse>(`/generation/status/${taskId}`)
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  cancelTask: async (taskId: number): Promise<void> => {
    try {
      await apiClient.delete(`/generation/cancel/${taskId}`)
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  getQueue: async (statusFilter?: string): Promise<TaskResponse[]> => {
    try {
      const params = statusFilter ? { status_filter: statusFilter } : {}
      const response = await apiClient.get<TaskResponse[]>('/generation/queue', { params })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  translate: async (text: string, provider?: string): Promise<TranslateResponse> => {
    try {
      const response = await apiClient.post<TranslateResponse>('/llm/translate', {
        text,
        provider,
      })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  enhance: async (prompt: string, provider?: string): Promise<EnhanceResponse> => {
    try {
      const response = await apiClient.post<EnhanceResponse>('/llm/enhance', {
        prompt,
        provider,
      })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },
}
