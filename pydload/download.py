import time
import uuid
import logging
import requests
import progressbar

gb = 1024 * 1024 

def dload(url=None, urls=None, save_to_path=None, timeout=10, max_time=30, verbose=True):
    '''

    Parameters:

    url (str): URL of the file to be downloaded.
    urls (list): Ordered list of URLs to be downloaded as a single file.

    save_to_path (str): Save as. If not provided, will be saved in the working directory with file_name auto identified from url.

    timeout (int): timeout for the initial handshake for requests.

    max_time (int): Kill the download if it takes more than max_time seconds.

        # Useful when you don't know the size of files before hand and don't want to download very large files.
    
    verbose (bool default:True): self explanatory


    Returns:

    False if downloading failed or stopped based on max_time. file_path if download is successful.

    '''
    if url and urls and url != urls:
        print("Only one of url or urls should be supplied")
        return
        
    if url and not urls:
        urls = [url]
    if isinstance(url, list):
        urls = [i for i in url]
    
    if not isinstance(urls, list):
        print("urls should be a list")
        return
    
    for i, url in enumerate(urls):
        url = url.rstrip('/')
        if 'http://' not in url[:7] and 'https://' not in url[:8]:
            if verbose:
                print('Assuming http://')
            url = 'http://' + url
        
        urls[i] = url

    if not save_to_path:
        url = urls[0]
        save_to_path = url.split('/')[-1].split('?')[0]
        if not save_to_path.strip():
            save_to_path = url.split('/')[-2]

        if not save_to_path.strip():
            save_to_path = str(uuid.uuid4())
            if verbose:
                print('Saving file as', save_to_path)

        if verbose:
            print('Saving the file at', save_to_path)

    if max_time:
        if verbose:
            print("The download will be auto-terminated in", max_time, "if not completed.")

    f = open(save_to_path, 'wb')
    start_time = time.time()

    for url in urls:
        try:
            request = requests.get(url, timeout=timeout, stream=True, verify=True, allow_redirects=True)
        except:
            if verbose:
                print('SSL certificate not verified...')
            request = requests.get(url, timeout=timeout, stream=True, verify=False, allow_redirects=True)

        file_size = None
        try:
            file_size = (float(request.headers['Content-length'])// gb) + 1
        except:
            if verbose:
                print('Content-length not found, file size cannot be estimated.')
            pass

        is_stopped = False


        if verbose:
            for chunk in progressbar.progressbar(request.iter_content(gb), max_value=file_size, prefix='GB'):
                f.write(chunk)
                if max_time:
                    if time.time() - start_time >= max_time:
                        is_stopped = True
                        break
        
        else:
            for chunk in request.iter_content(gb):
                f.write(chunk)
                if max_time:
                    if time.time() - start_time >= max_time:
                        is_stopped = True
                        break
        
        if is_stopped:
            if verbose:
                print('Stopped due to excess time')
            return False
    
    else:
        if verbose:
            print('Succefully Downloaded to:', save_to_path)
        return save_to_path


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='CLI for pydload')

    parser.add_argument('url', type=str, help='URL of the file to be downloaded.')

    parser.add_argument('save_to_path', type=str, nargs='?', help='save as file path/name')

    parser.add_argument('--max_time', type=int, help='Maximum time to be spent on download')
    parser.add_argument('--timeout', type=int, help='Reuest timeout')

    args = parser.parse_args()

    url = args.url
    save_to_path = args.save_to_path
    max_time = args.max_time
    timeout = args.timeout
    if not timeout: timeout=10

    dload(url, save_to_path=save_to_path, timeout=timeout, max_time=max_time, verbose=True)


if __name__ == '__main__':
    cli()
