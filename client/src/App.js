import React, { Component } from 'react';
import logo from './bars.svg';
import './App.css';
import axios from 'axios';
import settings from './settings.json';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { value: '' };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({ value: event.target.value });
  }

  handleSubmit(event) {
    console.log('A value was submitted: ' + this.state.value);
    event.preventDefault();
    let spacing = 2;
    axios.get(settings.server_url)
      .then(response => console.log('A response was received: ', JSON.stringify(response, null, spacing)))
  }

  render() {
    return (
      <div className="App">

        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to Musically</h1>
        </header>

        <form className="input" onSubmit={this.handleSubmit}>
          <label>
            Send a value to the server:
            <p />
            <input type="text" value={this.state.value} onChange={this.handleChange} />
          </label>
          <p />
          <input type="submit" value="Go" />
        </form>

      </div>
    );
  }
}

export default App;
