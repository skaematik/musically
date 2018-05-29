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
      .then(response => {
        console.log(response);
        console.log(JSON.parse(response.data.to_be_sent));
        console.log('A response was received: ', JSON.stringify(response, null, spacing));
      }
    );
  }

  componentDidMount() {
    let osmd = new window.opensheetmusicdisplay.OpenSheetMusicDisplay("osmd");
    // let xml = "http://downloads2.makemusic.com/musicxml/MozaVeilSample.xml";
    let xml = "MozaVeilSample.xml";
    osmd.load(xml).then(
      function () {
        osmd.render();
      }
    );
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

        <p />

        <div id="osmd"></div>

      </div>
    );
  }
}

export default App;
