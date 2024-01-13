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
                <div className="info-container">
                  <div className="user-info">
                    <h2>User Information</h2>
                    <table>
                      <tbody>
                      <tr>
                        <th>ID</th>
                        <td>{userData.twitch_id}</td>
                      </tr>
                      <tr>
                        <th>Created At</th>
                        <td>{userData.created_at && new Date(userData.created_at).toLocaleDateString()}</td>
                      </tr>
                      <tr>
                        <th>Affiliated</th>
                        <td>{userData.affiliated ? 'Yes' : 'No'}</td>
                      </tr>
                      <tr>
                        <th>Language</th>
                        <td>{userData.language}</td>
                      </tr>
                      <tr>
                        <th>Mature Content</th>
                        <td>{userData.mature ? 'Yes' : 'No'}</td>
                      </tr>
                      <tr>
                        <th>Last Updated</th>
                        <td>{userData.updated_at && new Date(userData.updated_at).toLocaleDateString()}</td>
                      </tr>
                      </tbody>
                    </table>
                  </div>

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

                {userData.algorithm === 'Popularity' && userData.results && (
                    <div className="recommendations">
                      <h2>Recommendations by Popularity</h2>
                      <table>
                        <thead>
                        <tr>
                          <th>Node</th>
                          <th>Affiliate</th>
                          <th>Language</th>
                          <th>Mature</th>
                          <th>Created At</th>
                          <th>Updated At</th>
                          <th>Num. Edges</th>
                        </tr>
                        </thead>
                        <tbody>
                        {userData.results.map((recommendation) => (
                            <tr key={recommendation.Node}>
                              <td>{recommendation.Node}</td>
                              <td>{recommendation.affiliate ? 'Yes' : 'No'}</td>
                              <td>{recommendation.language}</td>
                              <td>{recommendation.mature ? 'Yes' : 'No'}</td>
                              <td>{recommendation.created_at}</td>
                              <td>{recommendation.updated_at}</td>
                              <td>{recommendation.num_edges}</td>
                            </tr>
                        ))}
                        </tbody>
                      </table>
                    </div>
                )}

                {userData.algorithm === 'Link prediction' && userData.results && (
                    <div className="recommendations">
                      <h2>Recommendations by Link Prediction</h2>
                      <table>
                        <thead>
                        <tr>
                          <th>Node</th>
                          <th>Affiliate</th>
                          <th>Language</th>
                          <th>Mature</th>
                          <th>Created At</th>
                          <th>Updated At</th>
                          <th>Probability</th>
                          {/*<th>Predict. Score</th>*/}
                          <th>Comm.</th>
                        </tr>
                        </thead>
                        <tbody>
                        {userData.results.map((recommendation) => (
                            <tr key={recommendation.Node}>
                              <td>{recommendation.Node}</td>
                              <td>{recommendation.affiliate ? 'Yes' : 'No'}</td>
                              <td>{recommendation.language}</td>
                              <td>{recommendation.mature ? 'Yes' : 'No'}</td>
                              <td>{new Date(recommendation.created_at).toLocaleDateString()}</td>
                              <td>{new Date(recommendation.updated_at).toLocaleDateString()}</td>
                              <td>{recommendation.LinkProbability.toFixed(3)}</td>
                              {/*<td>{recommendation.LinkPredScore.toFixed(3)}</td>*/}
                              <td>{recommendation.Community_rs}</td>
                            </tr>
                        ))}
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
