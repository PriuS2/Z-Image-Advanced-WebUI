import axios from 'axios'

const API_BASE_URL = '/api'

interface LoginResponse {
  access_token: string
  token_type: string
}

interface User {
  id: number
  username: string
  settings: Record<string, unknown>
}

export const authApi = {
  login: async (username: string, password: string): Promise<LoginResponse> => {
    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)

    const response = await axios.post<LoginResponse>(
      `${API_BASE_URL}/auth/login`,
      formData,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      }
    )
    return response.data
  },

  register: async (username: string, password: string): Promise<User> => {
    const response = await axios.post<User>(`${API_BASE_URL}/auth/register`, {
      username,
      password,
    })
    return response.data
  },

  getMe: async (token: string): Promise<User> => {
    const response = await axios.get<User>(`${API_BASE_URL}/auth/me`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
    return response.data
  },
}
