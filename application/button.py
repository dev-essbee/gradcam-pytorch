def create_button():
    github_repo_url = 'https://github.com/dev-essbee/gradcam-pytorch'

    button_html = f'''
    <a href="{github_repo_url}" target="_blank" style="text-decoration: none;">
        <button style="
            background-color: #000000; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            font-size: 16px; 
            cursor: pointer; 
            display: flex; 
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            box-shadow: 0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12);
            transition: background-color 0.3s, box-shadow 0.3s;
        " onmouseover="this.style.backgroundColor='#333333'; this.style.boxShadow='0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 4px 0px rgba(0,0,0,0.14), 0px 1px 8px 0px rgba(0,0,0,0.12)';" onmouseout="this.style.backgroundColor='#000000'; this.style.boxShadow='0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12)';">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="height: 20px; margin-right: 8px;">
            Fork on GitHub
        </button>
    </a>
    '''

    return button_html

