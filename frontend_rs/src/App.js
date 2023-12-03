import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [username, setUsername] = useState('');
  const [userData, setUserData] = useState(null);
  const [error, setError] = useState('');

  const fetchUserData = async () => {
    try {
      const response = await axios.get(`http://localhost:8000/fetch-twitch-data/${username}/`);
      setUserData(response.data);
      setError('');
    } catch (err) {
      setError('Failed to fetch data');
      setUserData(null);
    }
  };

  const clearData = () => {
    setUsername(''); // Clear the username input
    setUserData(null); // Clear the displayed user data
    setError(''); // Clear any displayed error
  };

  return (
      <div className="App">
        <header className="App-header">
          <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter Twitch Username"
          />
          <button onClick={fetchUserData}>Fetch User Data</button>
          <button onClick={clearData} className="clear">Clear</button>

          {userData && (
              <div className="user-data">
                <p><strong>ID:</strong> {userData.twitch_id}</p>
                <p><strong>Created At:</strong> {userData.created_at && new Date(userData.created_at).toLocaleDateString()}</p>
                <p><strong>Affiliated:</strong> {userData.affiliated ? 'Yes' : 'No'}</p>
                <p><strong>Language:</strong> {userData.language}</p>
                <p><strong>Mature Content:</strong> {userData.mature ? 'Yes' : 'No'}</p>
                <p><strong>Last Updated:</strong> {userData.updated_at && new Date(userData.updated_at).toLocaleDateString()}</p>

                {userData.community !== undefined && (
                    <div className="community-data">
                      <h2>Community Prediction</h2>
                      <table>
                        <tbody>
                        <tr>
                          <th>Community</th>
                          <td>{userData.community}</td>
                        </tr>
                        <tr>
                          <th>Probability</th>
                          <td>{userData.probability}</td>
                        </tr>
                        </tbody>
                      </table>
                    </div>
                )}
              </div>
          )}

          {error && <p className="error">{error}</p>}
        </header>
      </div>
  );
}

export default App;
